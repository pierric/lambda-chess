{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE TemplateHaskell       #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
module Fei.AI.MCTS where

import           Control.Lens       (ix, makeLenses, (+~))
import           Control.Zipper
import           Data.Constraint
import           Data.Monoid        (First (..))
import           RIO
import           RIO.State
import qualified RIO.Vector         as V
import qualified RIO.Vector.Partial as V


cSCALAR, cEPSILON :: Float
cSCALAR = 1 / sqrt 2.0
cEPSILON = 1e-4

data Node a =
    Node { _node_v        :: !a
         , _node_i        :: !Int
         , _node_q        :: !Float
         , _node_n        :: !Int
         , _node_children :: !(Vector (Node a))
         }
    deriving (Show, Generic, NFData)

makeLenses ''Node

type Succ m a v = Cursor a -> m (Vector (Node a, v))
type Choose m a v = Vector (Node a, v) -> m Int
type Judge a = a -> Float
type Precondition a = Cursor a -> Maybe Float

(<||>) :: Precondition a -> Precondition a -> Precondition a
cond1 <||> cond2 = \c -> let v1 = cond1 c :: Maybe Float
                             v2 = cond2 c :: Maybe Float
                          in v1 <|> v2

class UpwardeNavigable z a where
    type UpwardZipper z a
    parent :: z :>> Node a
           -> Maybe (UpwardZipper z a :>> Node a,
                     Dict (UpwardeNavigable (UpwardZipper z a) a,
                           Zipping (UpwardZipper z a) (Node a),
                           Zipped (UpwardZipper z a) (Node a) ~ Node a))

instance UpwardeNavigable Top a where
    type UpwardZipper Top a = Top
    parent z = Nothing

instance (UpwardeNavigable z a, Zipping z (Node a), Zipped z (Node a) ~ Node a) => UpwardeNavigable (z :>> Node a) a where
    type UpwardZipper (z :>> Node a) a = z
    parent z = let pz = z & upward
                in pz `seq` Just (pz, Dict)

data Cursor a where
    Cursor :: (UpwardeNavigable z a, Zipping z (Node a), Zipped z (Node a) ~ Node a)
           => !(z :>> Node a) -> Cursor a

type Root a = Top :>> Node a

uct logn c =
    let n = fromIntegral (c ^. node_n) + cEPSILON
        average_reward = c ^. node_q / n
        exploration = 2 * cSCALAR * sqrt (2 * logn / n)
     in average_reward + exploration

childAt :: Int -> z :>> Node a -> z :>> Node a :>> Node a
childAt n =
    -- this loses the index to the children
    -- `tooth` would return always 0
    fromWithin (node_children . ix n)
    {-
     - zipper can track the index, which can be retrieved by `tooth`, automatically in this way,
     - HOWEVER, it is significantly slower, and a x3 memory overhead.
     -
    case z & fromWithin (node_children . traverse) & moveTo n of
      Just z' -> z'
      Nothing -> error $ "no child at index " ++ show n
     -}

expand :: MonadIO m => Succ m a v -> Cursor a -> m (Cursor a)
expand succ cur@(Cursor z) = do
    children <- succ cur <&> V.map fst
    return $ Cursor $ z & focus . node_children %~ (V.++ children)

select :: (HasCallStack, Monad m) => Cursor a -> StateT Int m (Cursor a)
select c@(Cursor z)
  | V.null (cur_node ^. node_children) = return c
  | otherwise = do
      let logn = log $ fromIntegral (cur_node ^. node_n) + cEPSILON
          choice = V.maxIndex $ V.map (uct logn) $ cur_node ^. node_children
      modify (+1)
      select (Cursor $ childAt choice z)
    where
        cur_node = z & view focus

simulate :: (MonadIO m, Show a)
         => Succ m a v
         -> Choose m a v
         -> Judge a
         -> Precondition a
         -> Cursor a
         -> m Float
simulate succ choose judge pre_cond = go
    where
        go cur@(Cursor z) = do
            case pre_cond cur of
              Just outcome -> return outcome
              Nothing      -> do
                !space <- succ cur
                if V.null space
                then return $ judge $ z & view (focus . node_v)
                else do
                    let z_upd = z & focus . node_children .~ V.map fst space
                    n <- choose space
                    go $ Cursor $ childAt n z_upd

backward :: Int -> Cursor a -> Float -> Cursor a
backward = go
    where
      upd z reward = z & focus . node_n +~ 1 & focus . node_q +~ reward
      go :: Int -> Cursor a -> Float -> Cursor a
      go 0 (Cursor z) reward = uncurry descendTo $ upToRoot (Cursor z) reward []
      go n (Cursor z) reward = case parent (upd z reward) of
                                 Nothing -> error "the backward pass goes beyond the root"
                                 Just (!pz, Dict) -> go (n-1) (Cursor pz) (-reward)
      -- stand at the last stop, and go up/update to the root
      -- return the path and standing at root finally
      upToRoot :: Cursor a -> Float -> [Int] -> ([Int], Cursor a)
      upToRoot (Cursor z) reward path =
          let index = z & view (focus . node_i)
              z' = upd z reward
           in case parent z' of
                Nothing         -> (path, Cursor z')
                Just (pz, Dict) -> upToRoot (Cursor pz) (-reward) (index:path)

      -- descend according the given path
      -- NOTE: saveTap/restoreTape seems to have a memory leak, therefore
      -- we make a manuel bookmarking and descending
      descendTo :: [Int] -> Cursor a -> Cursor a
      descendTo [] cur              = cur
      descendTo (i:path) (Cursor z) = descendTo path (Cursor $ childAt i z)

mcts :: (HasCallStack, MonadIO m, Show a)
     => Succ m a v -> Choose m a v -> Judge a -> Precondition a
     -> Int -> Cursor a -> m (Maybe (Int, Cursor a))
mcts succ choose judge pre_cond = go
    where
        expected_reward n = n ^. node_q / (fromIntegral (n ^. node_n) + cEPSILON)
        go 0 root@(Cursor z) = do
            let children = z & view (focus . node_children)
            if V.null children
              then return Nothing
              else do
                let best = V.maxIndex $ V.map expected_reward children
                return $ Just (best, root)
        go n root = do
            (cur_at_leaf, path_length) <- flip runStateT 0 $ select root

            -- once we selected a path that not satifying pre_cond,
            -- we wouldn't update the q/n any more. Therefore it can simply
            -- jumps to the final loop step.
            case pre_cond cur_at_leaf of
              Just _  -> go 0 root
              Nothing -> do
                cur_expanded <- expand succ cur_at_leaf
                reward <- simulate succ choose judge pre_cond cur_expanded
                let root_updated = backward path_length cur_expanded reward
                go (n - 1) root_updated

