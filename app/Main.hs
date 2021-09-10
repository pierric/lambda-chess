{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}
module Main where

import           Control.Lens                 (_2, ix, use, (^?!))
import           Control.Zipper               (rezip, zipper)
import qualified Data.Binary                  as Binary (encodeFile)
import qualified Data.Conduit                 as C (runConduit, (.|))
import qualified Data.Conduit.Combinators     as C (sinkFile)
import qualified Data.Conduit.List            as C (chunksOf, map, mapM)
import qualified Data.Store                   as Store
import qualified Data.Yaml                    as Yaml
import           Formatting                   (formatToString, int, sformat,
                                               stext, (%))
import           GHC.TypeLits                 (KnownSymbol)
import           Game.Chess                   (Ply, color, opponent, startpos,
                                               toUCI)
import           RIO                          hiding (Const)
import           RIO.Directory                (createDirectoryIfMissing)
import           RIO.FilePath                 ((</>))
import qualified RIO.HashMap                  as M
import qualified RIO.HashSet                  as S
import           RIO.List                     (unzip3)
import qualified RIO.State                    as ST
import qualified RIO.Vector.Boxed             as VB
import qualified RIO.Vector.Boxed.Partial     as VB (head)
import qualified RIO.Vector.Boxed.Unsafe      as VB
import qualified RIO.Vector.Storable          as VS
import qualified RIO.Vector.Storable.Partial  as VS
import qualified RIO.Vector.Storable.Unsafe   as VS
import           System.Random.Stateful

import           Fei.AI.Chess
import           Fei.AI.MCTS
import           Immutable.Shuffle            (shuffle)
import           MXNet.Base
import           MXNet.Base.Operators.Tensor  as T
import           MXNet.Base.Profiler
import qualified MXNet.Base.Tensor.Functional as F
import           MXNet.NN
import           MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.Initializer         as I

cINPUTSHAPE = [1, 8, 8, 105]
cROLLOUT = 100
cTRAINBATCHSIZE = 4

playStep :: (HasCallStack,
            RandomGen g,
            HasLogFunc env,
            MonadThrow m,
            MonadReader env m,
            KnownSymbol t,
            HasSessionRef env (TaggedModuleState Float t),
            Session sess (TaggedModuleState Float t),
            MonadIO m)
         => IOGenM g
         -> Int
         -> Int
         -> Root BoardState
         -> m (Root BoardState, [Ply], ConduitData m (NDArray Float, NDArray Float, Float))
playStep randgen n_rollout step_index root = do
    logInfo . display $ sformat ("[Play iter " % int % "]") step_index

    -- liftIO $ setConfig (#filename := "/mnt/hdd0/all.prof" .& #profile_all := True
    --                  .& #aggregate_stats := True .& Nil)

    (cur, plys) <-
            if step_index < 0
          then play root succPositions (uniformlyChoose randgen) checkDraw n_rollout
          else do
              infr_sym <- runLayerBuilder (modelDef False)
              let inputs_shape = M.fromList [("inp", cINPUTSHAPE)]
              askSession $ do
                  withSharedParameters infr_sym inputs_shape $ \forward ->
                      play root
                           (succPositionsWithModel randgen forward)
                           (chooseByV randgen)
                           checkDraw
                           n_rollout

    -- liftIO (stats False Table Total Descending) >>= logInfo . display
    let root_new = case cur of Cursor z -> zipper $ force $ rezip z
    return (root_new, plys, ConduitData (Just 1) (encodeForTraining cur))

    where
    {-# SCC checkDraw #-}
    checkDraw = threeFoldDraw (Just 100) <||> fiftyMovesDraw
    --checkDraw = threeFoldDraw (Just 200)
    --checkDraw = fiftyMovesDraw

    {-# SCC succPositionsWithModel #-}
    succPositionsWithModel :: (MonadIO m, RandomGen g)
                           => IOGenM g
                           -> (HashMap Text (NDArray Float) -> IO [NDArray Float])
                           -> Cursor BoardState
                           -> m (VB.Vector (Node BoardState, Float))
    succPositionsWithModel randgen forward cur = liftIO $ do
        -- get allowed Ply from all_pos
        all_pos     <- succPositions cur

        if VB.null all_pos then
            return VB.empty
        else do
            mc <- randomRM (0 :: Int, 9) randgen
            if mc > 2 then
                return $ VB.map (_2 .~ 0) all_pos
            else do
                -- make prediction
                input  <- encodeForInference cur contextPinnedCPU >>= F.expandDims 0
                output <- forward $ M.fromList [("inp", input)]

                distr_logits <- F.squeeze Nothing $ output ^?! ix 0
                -- distr_logits <- toCPU distr_logits

                let enc (node, _) = let pos  = node ^. node_v . board_position
                                        -- the ply was done in prior to the node (i.e. the opponent)
                                        cur_color = opponent $ color pos
                                        mply = node ^. node_v . board_ply
                                     in case mply >>= encodePly cur_color of
                                          Just plyidx -> fromIntegral plyidx
                                          Nothing     -> error $ "cannot encode the Ply: " ++ show mply

                all_pos     <- applyIOGen (shuffle all_pos) randgen
                let device = contextGPU0 -- contextCPU
                all_pos_enc <- makeNDArray [VB.length all_pos] device $ VS.convert $ VB.map enc all_pos

                -- -- look them up in the predicated distribution logits
                -- -- return all options
                -- distr_logits_valid <- F.takeI all_pos_enc distr_logits >>= toVector
                -- return $ VB.zip (VB.unzip all_pos & fst) (VB.convert distr_logits_valid)

                -- -- take the best k options
                -- distr_logits_valid <- F.takeI all_pos_enc distr_logits
                -- let k = min 10 (VB.length all_pos)
                -- (values, indices) <- F.topkBoth (Just 0) k distr_logits_valid
                -- indices <- VB.convert <$> toVector indices
                -- values  <- VB.convert <$> toVector values
                -- let nodes = VB.unzip all_pos & fst
                --     nodes_k = VB.map (\i -> nodes ^?! ix (truncate i)) indices
                -- return $ VB.zip nodes_k values

                -- return the best option. The value is set to dummy 0, since there is only one option
                distr_logits_valid <- F.takeI all_pos_enc distr_logits

                index <- F.argmax distr_logits_valid (Just 0) False >>= toScalar

                let choice = all_pos ^?! ix (fromIntegral index) & fst & node_i .~ 0
                return $ VB.singleton (choice, 0)

    {-# SCC chooseByV #-}
    chooseByV :: (MonadIO m, RandomGen g)
              => IOGenM g
              -> Choose m BoardState Float
    chooseByV randgen options =
        -- take the all nodes with maximum v
        -- random choose one from them
        --
        if VB.length options > 1 then
            uniformlyChoose randgen options
        else
            return 0

        -- options <- VB.unsafeThaw options
        -- VB.sortBy (flip compare `on` snd) options
        -- options <- VB.unsafeFreeze options
        -- let e0 = VB.head options
        --     !best_options = VB.takeWhile (\a -> snd a == snd e0) options
        -- if VB.length best_options > 1
        --     then uniformlyChoose randgen best_options
        --     else return $ fst $ VB.head best_options


batchify :: MonadIO m
         => Int
         -> ConduitData m (NDArray Float, NDArray Float, Float)
         -> ConduitData m (NDArray Float, NDArray Float, NDArray Float)
batchify batch_size (ConduitData (Just 1) dat) = ConduitData (Just batch_size) (batched dat)
    where
        batched dat = dat C..| C.chunksOf batch_size C..| C.mapM (liftIO . concat)
        concat :: [(NDArray Float, NDArray Float, Float)]
               -> IO (NDArray Float, NDArray Float, NDArray Float)
        concat vec = do
            let (a, b, c) = unzip3 vec
            na <- F.stack 0 a
            nb <- F.stack 0 b
            nc <- let vec = VS.fromList c
                   in fromVector [VS.length vec, 1] vec
            return (na, nb, nc)


trainStep :: (HasCallStack, Optimizer opt,
                Dataset d, MonadIO m, DatasetMonadConstraint d m,
                HasSessionRef env (TaggedModuleState Float "lambda-chess"),
                HasLogFunc env, MonadReader env m)
          => opt Float -> d m (NDArray Float, NDArray Float, NDArray Float) -> m ()
trainStep optm dat = do
    logInfo . display $ sformat "[Train] "

    let distr_loss   = Loss (Just "distr_ce")    (\p -> p ^?! ix 0)
        outcome_loss = Loss (Just "outcome_mse") (\p -> p ^?! ix 1)
    metrics <- newMetric "train" (distr_loss :* outcome_loss :* MNil)

    void $ forEachD_i dat $ \(i, (inp, distr, outcome)) -> askSession $ do
        let mapping = M.fromList [("inp", inp), ("distr", distr), ("outcome", outcome)]
        fitAndEval optm mapping metrics

        --exec <- use $ untag . mod_executor
        --liftIO $ do
        --    [out0, out1, out2, _] <- execGetOutputs exec
        --    out0 <- toVector out0
        --    out1 <- toVector out1
        --    traceShowM ("loss", out0, out1)

        --    void $ do
        --        out2 <- toCPU out2
        --        printNorm ("A", ParameterV out2)
        --        printNorm ("B", ParameterV distr)
        --        pred <- F.logSoftmax out2 1 Nothing
        --        printNorm ("C", ParameterV pred)
        --        loss <- F.mulNoBroadcast pred distr
        --        printNorm ("D", ParameterV loss)
        --        loss <- F.sum_ loss (Just [1]) True >>= F.rsubScalar 0
        --        printNorm ("E", ParameterV loss)
        --        loss <- F.mean loss Nothing False
        --        printNorm ("F", ParameterV loss)

        -- p <- use (untag . mod_params)
        -- mapM_ printNorm (M.toList p)

        when (i `mod` 10 == 0) $ do
            eval <- metricFormat metrics
            logInfo . display $ sformat (int % " " % stext) i eval

    -- where
    --     printNorm (name, param) = liftIO $ do
    --         a <- pure $ case param of
    --             ParameterV a   -> a
    --             ParameterG a _ -> a
    --             ParameterA a   -> a
    --         g <- pure $ case param of
    --             ParameterG _ g -> Just g
    --             _              -> Nothing

    --         let norm a = prim _norm (#data := a .& #ord := 1 .& Nil) >>= toVector <&> VS.head
    --         a <- norm a
    --         g <- mapM norm g
    --         traceShowM (name, a, g)

modelDef :: DType a => Bool -> Layer (Symbol a)
modelDef training = do
    inp     <- variable "inp"
    distr   <- variable "distr"
    outcome <- variable "outcome"

    sequential "features" $ do
        inp_chw <- F.transpose inp [0, 3, 1, 2]
        x <- convolution   (#data := inp_chw .& #kernel := [3,3] .& #num_filter := 400 .& Nil)
        x <- F.activation  (#data := x .& #act_type := #tanh .& Nil)
        x <- F.pooling     (#data := x .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        x <- convolution   (#data := x .& #kernel := [3,3] .& #num_filter := 1600 .& Nil)
        x <- F.activation  (#data := x .& #act_type := #tanh .& Nil)
        x <- F.pooling     (#data := x .& #kernel := [2,2] .& #pool_type := #max .& Nil)

        x <- F.flatten x

        distr_pred <- fullyConnected (#data := x .& #num_hidden := 4672 .& Nil)
        outcome_pred <- fullyConnected (#data := x .& #num_hidden := 1 .& Nil)
        outcome_pred <- F.activation (#data := outcome_pred .& #act_type := #sigmoid .& Nil)

        distr_logits_out <- blockGrad distr_pred
        outcome_pred_out <- blockGrad outcome_pred

        if not training
        then group [distr_logits_out, outcome_pred_out]
        else do
            distr_loss   <- sequential "softmax" $
                            F.softmaxCE 1 distr_pred distr Nothing >>= asLoss
            outcome_loss <- sequential "mse" $ do
                                t <- F.subNoBroadcast outcome_pred outcome
                                t <- F.square_ t
                                t <- F.mean t Nothing False
                                asLoss t
            group [distr_loss, outcome_loss, distr_logits_out, outcome_pred_out]
    where
        asLoss t = F.reshape [1] t >>= flip makeLoss 1.0

saveSnapshot :: (HasCallStack,
                MonadIO m,
                HasSessionRef env (TaggedModuleState Float "lambda-chess"),
                HasLogFunc env, MonadReader env m)
             => Int -> [Ply] -> m ()
saveSnapshot step plys = do
    let dir = "snapshots"
        play_file  = formatToString ("play_"  % int % ".yaml")   step
        model_file = formatToString ("model_" % int % ".state") step
    createDirectoryIfMissing False dir
    liftIO $ Yaml.encodeFile (dir </> play_file) (map toUCI plys)
    askSession $ saveState (step == 0) (dir </> model_file)

main = do
    randgen <- newIOGenM $ mkStdGen 22
    runFeiM $ Simple $ do
        model <- runLayerBuilder (modelDef True)
        initSession @"lambda-chess" model (Config {
            _cfg_data = M.fromList [
                ("inp", cINPUTSHAPE),
                ("distr", [1, 4672]),
                ("outcome", [1, 1])
            ],
            _cfg_label = [],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = SomeInitializer default_initializer,
            _cfg_fixed_params = S.fromList [],
            _cfg_context = contextGPU0 })

        logInfo "load weights"
        askSession $ loadState "snapshots/weights" ["inp", "distr", "outcome"]

        optm <- makeOptimizer SGD'Mom (Const 0.001) Nil

        let root = zipper $ Node (BoardState startpos Nothing) 0 0 0 VB.empty
        void $ flip ST.evalStateT root $ do
            forM_ [0..100] $ \step -> do
                cur <- ST.get
                (!cur, plys, dat) <- lift $ playStep randgen cROLLOUT step cur
                ST.put cur
                lift $ trainStep optm (batchify cTRAINBATCHSIZE dat)
                lift $ saveSnapshot step plys
                -- -- dump the dataset for debug
                -- ConduitData _ cc <- playStep randgen 100 step
                -- let name = formatToString ("data." % int % ".binary") step
                --     sink = C.sinkFile name
                -- C.runConduit $ cc C..| C.map Store.encode C..| sink

    where
        default_initializer :: I.CustomInit Float
        default_initializer = I.CustomInit $ \name arr -> do
            shp <- ndshape arr
            case length shp of
                3 -> initNDArray (I.InitXavier 2.0 I.XavierGaussian I.XavierIn) name arr
                _ -> initNDArray (I.InitNormal 0.1) name arr

