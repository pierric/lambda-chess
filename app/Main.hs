{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}
module Main where

import           Control.Lens                 (ix, (^?!))
import qualified Data.Conduit                 as C ((.|))
import qualified Data.Conduit.List            as C (chunksOf, mapM)
import qualified Data.Vector.Algorithms.Intro as VB (sortBy)
import           Formatting                   (int, sformat, stext, (%))
import           GHC.TypeLits                 (KnownSymbol)
import           Game.Chess                   (color)
import           RIO                          hiding (Const)
import qualified RIO.HashMap                  as M
import qualified RIO.HashSet                  as S
import           RIO.List                     (unzip3)
import qualified RIO.Vector.Boxed             as VB
import qualified RIO.Vector.Boxed.Partial     as VB (head)
import qualified RIO.Vector.Boxed.Unsafe      as VB
import qualified RIO.Vector.Storable          as VS
import qualified RIO.Vector.Storable.Unsafe   as VS
import           System.Random.Stateful

import           Fei.AI.Chess
import           Fei.AI.MCTS
import           MXNet.Base
import qualified MXNet.Base.Tensor.Functional as F
import           MXNet.NN
import           MXNet.NN.DataIter.Conduit
import qualified MXNet.NN.Initializer         as I


playStep :: (HasCallStack,
            RandomGen g,
            HasLogFunc env,
            MonadThrow m,
            MonadReader env m,
            KnownSymbol t,
            HasSessionRef env (TaggedModuleState Float t),
            Session sess (TaggedModuleState Float t),
            MonadIO m)
         => IOGenM g -> Int -> Int -> m (ConduitData m (NDArray Float, NDArray Float, Float))
playStep randgen n_rollout step_index = do
    logInfo . display $ sformat ("[Play " % int % "]") step_index
    (cur, plys) <-
        if step_index < 10
          then {-# SCC "play" #-} play succPositions (uniformlyChoose randgen) n_rollout
          else do
              infr_sym <- runLayerBuilder (modelDef False)
              let inputs_shape = M.fromList [("inp", [1, 105, 8, 8])]
              askSession $ do
                  withSharedParameters infr_sym inputs_shape $ \forward -> do
                      play (succPositionsWithModel forward) (chooseByV randgen) n_rollout
    return $ ConduitData (Just 1) (encodeForTraining cur)

    where
    succPositionsWithModel forward cur = do
        -- get allowed Ply from all_pos
        all_pos <- succPositions cur

        -- make prediction
        input   <- encodeForInference cur >>= F.expandDims 0
        output  <- forward $ M.fromList [("inp", input)]

        -- look them up in the predicated distribution logits
        distr_logits <- liftIO $ toVector $ output ^?! ix 0
        let lookup (node, _) = let pos  = node ^. node_v . board_position
                                   mply = node ^. node_v . board_ply
                                in case mply >>= encodePly (color pos) of
                                     Just plyidx -> let v = VS.unsafeIndex distr_logits plyidx
                                                     in (node, v)
                                     Nothing     -> error "cannot encode the Ply"
        return $ VB.map lookup all_pos

    chooseByV randgen options = do
        -- take the all nodes with maximum v
        -- random choose one from them
        options <- VB.unsafeThaw options
        VB.sortBy (flip compare `on` snd) options
        options <- VB.unsafeFreeze options
        let e0 = VB.head options
            best_options = VB.takeWhile (\a -> snd a == snd e0) options
        if VB.length best_options > 1
            then uniformlyChoose randgen best_options
            else return $ fst $ VB.head best_options


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

        when (i `mod` 20 == 0) $ do
            eval <- metricFormat metrics
            logInfo . display $ sformat (int % " " % stext) i eval

modelDef :: DType a => Bool -> Layer (Symbol a)
modelDef training = do
    inp     <- variable "inp"
    distr   <- variable "distr"
    outcome <- variable "outcome"

    sequential "features" $ do
        x <- convolution (#data := inp  .& #kernel := [5,5] .& #num_filter := 200 .& Nil)
        x <- F.activation  (#data := x .& #act_type := #tanh .& Nil)
        x <- F.pooling     (#data := x .& #kernel := [2,2] .& #pool_type := #max .& Nil)
        x <- F.flatten x

        distr_pred <- fullyConnected (#data := x .& #num_hidden := 4672 .& Nil)
        outcome_pred <- fullyConnected (#data := x .& #num_hidden := 1 .& Nil)
        outcome_pred <- F.activation (#data := outcome_pred .& #act_type := #tanh .& Nil)

        distr_logits_out <- blockGrad distr_pred
        outcome_pred_out <- blockGrad outcome_pred

        if not training
        then group [distr_logits_out, outcome_pred_out]
        else do
            distr_loss <- sequential "softmax" $
                          F.softmaxCE 1 distr_pred distr Nothing >>= F.reshape [1] >>= flip makeLoss 1.0
            outcome_loss <- F.subNoBroadcast outcome_pred outcome >>= F.square_ >>= flip makeLoss 1.0
            group [distr_loss, outcome_loss, distr_logits_out, outcome_pred_out]

main = do
    randgen <- newIOGenM $ mkStdGen 22
    runFeiM $ Simple $ do
        model <- runLayerBuilder (modelDef True)
        initSession @"lambda-chess" model (Config {
            _cfg_data = M.fromList [
                ("inp", [1, 105, 8, 8]),
                ("distr", [1, 4672]),
                ("outcome", [1, 1])
            ],
            _cfg_label = [],
            _cfg_initializers = M.empty,
            _cfg_default_initializer = SomeInitializer default_initializer,
            _cfg_fixed_params = S.fromList [],
            _cfg_context = contextGPU0 })

        optm <- makeOptimizer SGD'Mom (Const 0.001) Nil

        forM_ [0..1] $ \step -> do
            dat <- playStep randgen 100 step
            trainStep optm (batchify 4 dat)

    where
        default_initializer :: I.CustomInit Float
        default_initializer = I.CustomInit $ \name arr -> do
            shp <- ndshape arr
            case length shp of
                1 -> initNDArray I.InitZeros name arr
                2 -> initNDArray (I.InitXavier 2.0 I.XavierGaussian I.XavierIn) name arr
                _ -> initNDArray (I.InitNormal 0.1) name arr

