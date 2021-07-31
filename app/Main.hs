{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedLabels  #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections     #-}
{-# LANGUAGE TypeApplications  #-}
module Main where

import           Fei.AI.Chess
import           MXNet.Base
import           MXNet.NN
import           MXNet.NN.DataIter.Vec  (DatasetVector (..))
import qualified MXNet.NN.Initializer   as I

import           Control.Lens           (ix, (^?!))
import           Formatting             (int, sformat, stext, (%))
import           RIO                    hiding (Const)
import qualified RIO.HashMap            as M
import qualified RIO.HashSet            as S
import qualified RIO.Vector.Boxed       as VB
import           System.Random.Stateful

modelDef :: DType a => Layer (Symbol a)
modelDef = do
    inp     <- variable "inp"
    distr   <- variable "distr"
    outcome <- variable "outcome"

    sequential "features" $ do
        x <- convolution (#data := inp  .& #kernel := [5,5] .& #num_filter := 200 .& Nil)
        x <- activation  (#data := x .& #act_type := #tanh .& Nil)
        x <- pooling     (#data := x .& #kernel := [2,2] .& #pool_type := #max .& Nil)
        x <- flatten x

        distr_pred <- fullyConnected (#data := x .& #num_hidden := 4672 .& Nil)
        distr_loss <- softmaxCE 1 distr_pred distr Nothing >>= reshape [1] >>= flip makeLoss 1.0

        outcome_pred <- fullyConnected (#data := x .& #num_hidden := 1 .& Nil)
        outcome_pred <- activation (#data := outcome_pred .& #act_type := #tanh .& Nil)
        outcome_loss <- subNoBroadcast outcome_pred outcome >>= square_ >>= flip makeLoss 1.0
        outcome_pred <- blockGrad outcome_pred

        group [distr_loss, outcome_loss, outcome_pred]

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

playStep :: (HasCallStack, RandomGen g, HasLogFunc env, MonadReader env m, MonadIO m)
         => IOGenM g -> Int -> m [(NDArray Float, NDArray Float, Float)]
playStep randgen n_rollout = do
    logInfo . display $ sformat "[Play]"
    (cur, plys) <- liftIO $ play (uniformly_choose randgen) n_rollout
    prepareTraining cur

batchify :: MonadIO m
         => Int
         -> [(NDArray Float, NDArray Float, Float)]
         -> m (DatasetVector m (NDArray Float, NDArray Float, NDArray Float))
batchify batch_size dat = liftIO $ do
    batches <- mapM concat $ chunkOf batch_size $ VB.fromList dat
    return $ DatasetVector $ VB.fromList batches
    where
        chunkOf :: Int -> Vector a -> [Vector a]
        chunkOf size vec
          | VB.length vec < size = []
          | otherwise = let (b, r) = VB.splitAt size vec
                         in b : chunkOf size r
        concat :: Vector (NDArray Float, NDArray Float, Float)
               -> IO (NDArray Float, NDArray Float, NDArray Float)
        concat vec = do
            let (a, b, c) = VB.unzip3 vec
            na <- stack 0 $ VB.toList a
            nb <- stack 0 $ VB.toList b
            nc <- fromVector [VB.length c, 1] (VB.convert c)
            return (na, nb, nc)


main = do
    randgen <- newIOGenM $ mkStdGen 22
    runFeiM $ Simple $ do
        model <- runLayerBuilder modelDef
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

        playStep randgen 100 >>= batchify 4 >>= trainStep optm

    where
        default_initializer :: I.CustomInit Float
        default_initializer = I.CustomInit $ \name arr -> do
            shp <- ndshape arr
            case length shp of
                1 -> initNDArray I.InitZeros name arr
                2 -> initNDArray (I.InitXavier 2.0 I.XavierGaussian I.XavierIn) name arr
                _ -> initNDArray (I.InitNormal 0.1) name arr

