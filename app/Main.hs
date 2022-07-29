{-# LANGUAGE BangPatterns         #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE ImplicitParams       #-}
{-# LANGUAGE OverloadedLabels     #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin=Data.Record.Anon.Plugin#-}
module Main where

import           Control.Lens                 (_2, ix, use, (^?!))
import           Control.Zipper               (rezip, zipper)
import qualified Data.Binary                  as Binary (encodeFile)
import qualified Data.Conduit                 as C (runConduit, (.|))
import qualified Data.Conduit.Combinators     as C (sinkFile)
import qualified Data.Conduit.List            as C (chunksOf, map, mapM)
import           Data.Default.Class
import qualified Data.HashMap.Strict          as M
import           Data.Record.Anon.Simple      (empty)
import qualified Data.Store                   as Store
import qualified Data.Yaml                    as Yaml
import           Formatting                   (formatToString, int, sformat,
                                               stext, (%))
import           GHC.TypeLits                 (KnownSymbol)
import           Game.Chess                   (Ply, color, opponent, startpos,
                                               toUCI)
import           Immutable.Shuffle            (shuffle)
import           RIO                          hiding (Const)
import           RIO.Directory                (createDirectoryIfMissing)
import           RIO.FilePath                 ((</>))
import           RIO.List                     (unzip3)
import           RIO.NonEmpty                 (NonEmpty (..), (<|))
import           RIO.Partial                  (toEnum)
import qualified RIO.State                    as ST
import qualified RIO.Vector.Boxed             as VB
import qualified RIO.Vector.Boxed.Partial     as VB (head)
import qualified RIO.Vector.Boxed.Unsafe      as VB
import qualified RIO.Vector.Storable          as VS
import qualified RIO.Vector.Storable.Partial  as VS
import qualified RIO.Vector.Storable.Unsafe   as VS
import           System.Random.Stateful

import           Fei.AI.Chess
import           Fei.AI.MCTS                  hiding (backward)
import           MXNet.Base
import           MXNet.Base.AutoGrad
import           MXNet.Base.Operators.Tensor  as T
import           MXNet.Base.Profiler
import qualified MXNet.Base.Tensor.Functional as F
import           MXNet.NN.DataIter.Conduit
import           MXNet.NN.Initializer
import           MXNet.NN.LrScheduler
import           MXNet.NN.Module
import           MXNet.NN.Optimizer

cINPUTSHAPE = [1, 8, 8, 105]
cROLLOUT = 100
cTRAINBATCHSIZE = 4

playStep :: (HasCallStack,
            RandomGen g,
            HasLogFunc env,
            MonadThrow m,
            MonadReader env m,
            MonadIO m,
            ?device :: Context,
            Module mdl,
            ModuleDType mdl ~ Float,
            ModuleInput mdl ~ NDArray Float,
            ModuleOutput mdl ~ HashMap Text (NDArray Float))
         => IOGenM g
         -> Int
         -> Int
         -> Root BoardState
         -> mdl
         -> m (Root BoardState, [Ply], ConduitData m (NDArray Float, NDArray Float, Float))
playStep randgen n_rollout step_index root model = do
    logInfo . display $ sformat ("[Play iter " % int % "]") step_index

    -- liftIO $ setConfig (#filename := "/mnt/hdd0/all.prof" .& #profile_all := True
    --                  .& #aggregate_stats := True .& Nil)

    let (succ, choose)
            | step_index < 10 = (succPositions, uniformlyChoose randgen)
            | otherwise = (succPositionsWithModel randgen (forward model), chooseByV randgen)
    (cur, plys) <- liftIO $ recording False $ training False $
                   play root succ choose checkDraw n_rollout

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
                           -> (NDArray Float -> IO (HashMap Text (NDArray Float)))
                           -> Succ m BoardState Float
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
                output <- forward input
                distr_logits <- F.squeeze Nothing (output M.! "distr")
                distr_logits <- detach distr_logits
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

                index <- F.argmax distr_logits_valid (Just 0) False >>= toValue

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


trainStep :: (HasCallStack, Optimizer opt Float,
              Dataset d, MonadIO m, DatasetMonadConstraint d m,
              HasLogFunc env, MonadReader env m,
              ?device :: Context,
              Module mdl, ModuleDType mdl ~ Float,
              ModuleInput mdl ~ NDArray Float,
              ModuleOutput mdl ~ HashMap Text (NDArray Float))
          => opt Float
          -> d m (NDArray Float, NDArray Float, NDArray Float)
          -> mdl
          -> m ()
trainStep optm dat model = do
    logInfo . display $ sformat "[Train] "

    -- let distr_loss   = Loss (Just "distr_ce")    (\p -> p ^?! ix 0)
    --     outcome_loss = Loss (Just "outcome_mse") (\p -> p ^?! ix 1)
    -- metrics <- newMetric "train" (distr_loss :* outcome_loss :* MNil)

    void $ forEachD_i dat $ \(i, (inp, distr, outcome)) ->
        -- let mapping = M.fromList [("inp", inp), ("distr", distr), ("outcome", outcome)]
        -- fitAndEval optm mapping metrics
        liftIO $ recording True $ training True $ do
            out <- forward model inp
            let distr_pred = out M.! "distr"
                outcome_pred = out M.! "outcome"
            distr_loss   <- F.softmaxCE 1 distr_pred distr Nothing >>= asLoss
            outcome_loss <- do t <- F.subNoBroadcast outcome_pred outcome
                               t <- F.square_ t
                               t <- F.mean t Nothing False
                               asLoss t
            backward [distr_loss, outcome_loss] [] False False True

    where
        asLoss t = F.reshape [1] t
        --when (i `mod` 10 == 0) $ do
        --    eval <- metricFormat metrics
        --    logInfo . display $ sformat (int % " " % stext) i eval

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

data ChessModule u = ChessModule (Conv2D u) (Conv2D u) (Linear u) (Linear u)

paramPathPushScope :: Text -> ParameterPath -> ParameterPath
paramPathPushScope name (scope :> e) = name <| scope :> e

instance DType u => Module (ChessModule u) where
    type ModuleDType  (ChessModule u) = u
    type ModuleArgs   (ChessModule u) = ()
    type ModuleInput  (ChessModule u) = NDArray u
    type ModuleOutput (ChessModule u) = HashMap Text (NDArray u)

    init scope args user_init = do
        let conv_init = M.fromList
                [(scope :> ConvWeights,   initXavier 2.0 XavierGaussian XavierIn),
                 (scope :> ConvBias,      initNormal 0.1)]

            fc_init = M.fromList
                [(scope :> LinearWeights, initNormal 0.1),
                 (scope :> LinearBias,    initZeros) ]

            conv1_init = M.mapKeys (paramPathPushScope "conv1") conv_init
            conv2_init = M.mapKeys (paramPathPushScope "conv2") conv_init
            fc1_init   = M.mapKeys (paramPathPushScope "fc1")   fc_init
            fc2_init   = M.mapKeys (paramPathPushScope "fc2")   fc_init

        conv1  <- init ("conv1" <| scope)
                       (def {_conv_out_channels = 400, _conv_kernel = [3,3]})
                       (M.union user_init conv1_init)
        conv2  <- init ("conv2" <| scope)
                       (def {_conv_out_channels = 1600, _conv_kernel = [3,3]})
                       (M.union user_init conv2_init)
        fc1    <- init ("fc1" <| scope)
                       (LinearArgs 4672 True)
                       (M.union user_init fc1_init)
        fc2    <- init ("fc2" <| scope)
                       (LinearArgs 1 True)
                       (M.union user_init fc2_init)

        return (ChessModule conv1 conv2 fc1 fc2)

    forward (ChessModule conv1 conv2 fc1 fc2) inps = do
        inp_chw <- F.transpose inps [0, 3, 1, 2]
        x <- forward conv1 inp_chw
        x <- F.tanh x
        x <- F.maxPool x [2, 2] Nothing Nothing

        x <- forward conv2 x
        x <- F.tanh x
        x <- F.maxPool x [2, 2] Nothing Nothing

        x <- F.flatten x

        distr_pred   <- forward fc1 x

        outcome_pred <- forward fc2 x
        outcome_pred <- F.sigmoid outcome_pred

        return $ M.fromList [("distr", distr_pred), ("output", outcome_pred)]

    parameters (ChessModule conv1 conv2 fc1 fc2) = do
        p1 <- parameters conv1
        p2 <- parameters conv2
        p3 <- parameters fc1
        p4 <- parameters fc2
        return $ p1 `M.union` p2 `M.union` p3 `M.union` p4


-- saveSnapshot :: (HasCallStack,
--                 MonadIO m,
--                 HasSessionRef env (TaggedModuleState Float "lambda-chess"),
--                 HasLogFunc env, MonadReader env m)
--              => Int -> [Ply] -> m ()
-- saveSnapshot step plys = do
--     let dir = "snapshots"
--         play_file  = formatToString ("play_"  % int % ".yaml")   step
--         model_file = formatToString ("model_" % int % ".state") step
--     createDirectoryIfMissing False dir
--     liftIO $ Yaml.encodeFile (dir </> play_file) (map toUCI plys)
--     askSession $ saveState (step == 0) (dir </> model_file)

main = runSimpleApp $ do
    randgen <- newIOGenM $ mkStdGen 22

    let ?device = contextGPU0

    model <- liftIO $ (init ("model":|[]) () M.empty :: IO (ChessModule Float))
    optm  <- liftIO $ makeOptimizer (#sgd) (Const 0.001) empty

    let root = zipper $ Node (BoardState startpos Nothing) 0 0 0 VB.empty
    void $ flip ST.evalStateT root $ do
        forM_ [0..100] $ \step -> do
            cur <- ST.get
            (!cur, plys, dat) <- lift $ playStep randgen cROLLOUT step cur model
            ST.put cur
            lift $ trainStep optm (batchify cTRAINBATCHSIZE dat) model
            -- lift $ saveSnapshot step plys

                -- -- dump the dataset for debug
                -- ConduitData _ cc <- playStep randgen 100 step
                -- let name = formatToString ("data." % int % ".binary") step
                --     sink = C.sinkFile name
                -- C.runConduit $ cc C..| C.map Store.encode C..| sink

