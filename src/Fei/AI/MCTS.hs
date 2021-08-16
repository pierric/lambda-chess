{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
module Fei.AI.MCTS where

import           Control.Lens       (makeLenses, (+~))
import           Control.Zipper
import           Data.Constraint
import           RIO
import qualified RIO.Vector         as V
import qualified RIO.Vector.Partial as V
import           RIO.Writer

cSCALAR, cEPSILON :: Float
cSCALAR = 1 / sqrt 2.0
cEPSILON = 1e-4

data Node a =
    Node { _node_v        :: !a
         , _node_q        :: Float
         , _node_n        :: Int
         , _node_children :: !(Vector (Node a))
         }
    deriving Show

makeLenses ''Node

type Succ m a v = Cursor a -> m (Vector (Node a, v))
type Choose m a v = Vector (Node a, v) -> m (Node a)
type Judge a = a -> Float
type Precondition a = Cursor a -> Maybe Float

class UpwardeNavigable z a where
    type UpwardZipper z a
    parent :: z :>> Node a
           -> Maybe (UpwardZipper z a :>> Node a
                    ,Dict (UpwardeNavigable (UpwardZipper z a) a)
                    ,Dict (Zipping (UpwardZipper z a) (Node a))
                    ,Dict (Zipped (UpwardZipper z a) (Node a) ~ Node a))

instance UpwardeNavigable Top a where
    type UpwardZipper Top a = Top
    parent z = Nothing

instance (UpwardeNavigable z a, Zipping z (Node a), Zipped z (Node a) ~ Node a) => UpwardeNavigable (z :>> Node a :>> Vector (Node a)) a where
    type UpwardZipper (z :>> Node a :>> Vector (Node a)) a = z
    parent z = Just (z & upward & upward, Dict, Dict, Dict)

data Cursor a where
    Cursor :: (UpwardeNavigable z a, Zipping z (Node a), Zipped z (Node a) ~ Node a) => z :>> Node a -> Cursor a

uct logn c =
    let n = fromIntegral (c ^. node_n) + cEPSILON
        average_reward = c ^. node_q / n
        exploration = 2 * cSCALAR * sqrt (2 * logn / n)
     in average_reward + exploration

childAt :: Int -> Cursor a -> Cursor a
childAt n (Cursor c) = Cursor $
    c & downward node_children & fromWithin traverse & moveToward n

expand :: MonadIO m => Succ m a v -> Cursor a -> m (Cursor a)
expand succ cur@(Cursor z) = do
    children <- succ cur <&> V.map fst
    return $ Cursor $ z & focus . node_children %~ (V.++ children)

select :: (HasCallStack, Monad m) => Cursor a -> WriterT [Int] m (Cursor a)
select c@(Cursor z)
  | V.null (cur_node ^. node_children) = return c
  | otherwise = do
      let logn = log $ fromIntegral (cur_node ^. node_n) + cEPSILON
          choice = V.maxIndex $ V.map (uct logn) $ cur_node ^. node_children
      tell [choice]
      select (childAt choice c)
    where
        cur_node = z & view focus

simulate :: MonadIO m
         => Succ m a v
         -> Choose m a v
         -> Judge a
         -> Precondition a
         -> Cursor a
         -> m Float
simulate succ choose judge pre_cond cur@(Cursor z) = go cur
    where
        go cur@(Cursor z) = do
            case pre_cond cur of
              Just outcome -> return outcome
              Nothing -> do
                  space <- succ cur
                  if null space
                    then return $ judge $ view (focus . node_v) z
                    else do
                        n <- choose space
                        go $ Cursor $ zipper n

backward :: Int -> Cursor a -> Float -> Cursor a
backward n (Cursor z) reward =
    let cur_upd = Cursor (z & focus . node_n +~ 1 & focus . node_q +~ reward)
     in case n of
          0 -> cur_upd
          _ -> case cur_upd of
                 Cursor z ->
                     case parent z of
                       Nothing -> error "the backward pass goes beyond the root"
                       Just (pz, Dict, Dict, Dict) -> backward (n-1) (Cursor pz) (-reward)

mcts :: (HasCallStack, MonadIO m)
     => Succ m a v -> Choose m a v -> Judge a -> Precondition a
     -> Int -> Cursor a -> m (Maybe (Int, Cursor a))
mcts succ choose judge pre_cond = go
    where
        expected_reward n = n ^. node_q / (fromIntegral (n ^. node_n) + cEPSILON)
        go 0 root@(Cursor z) = do
            let children = z & downward node_children & view focus
            if V.null children
              then return Nothing
              else do
                let best = V.maxIndex $ V.map expected_reward children
                return $ Just (best, root)
        go n root = do
            (cur_at_leaf, path) <- runWriterT $ select root

            cur_expanded <- expand succ cur_at_leaf
            reward <- simulate succ choose judge pre_cond cur_expanded

            let root_updated = backward (length path) cur_expanded reward
            go (n - 1) root_updated

