{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE TemplateHaskell       #-}
module Fei.AI.Chess where

import           Control.Applicative
import           Control.Lens           (each, enum, from, makeLenses)
import           Control.Zipper
import           Data.Bifunctor         (bimap)
import           Data.Char
import           Data.Constraint        (Dict (..))
import           Game.Chess
import           RIO
import           RIO.List               (inits, scanr, unzip)
import           RIO.List.Partial       (head, init, tail)
import qualified RIO.NonEmpty           as NE
import           RIO.Partial            (fromJust, toEnum)
import qualified RIO.Vector             as V
import qualified RIO.Vector.Partial     as V
import qualified RIO.Vector.Storable    as VS
import           RIO.Writer
import           System.Random
import           System.Random.MWC
import           System.Random.Stateful (IOGenM, newIOGenM, uniformRM)

import           Fei.AI.MCTS
import           MXNet.Base
import           MXNet.Base.Tensor      as T

instance Enum PieceType where
    toEnum 0 = Pawn
    toEnum 1 = Knight
    toEnum 2 = Bishop
    toEnum 3 = Rook
    toEnum 4 = Queen
    toEnum 5 = King
    fromEnum Pawn   = 0
    fromEnum Knight = 1
    fromEnum Bishop = 2
    fromEnum Rook   = 3
    fromEnum Queen  = 4
    fromEnum King   = 5

newtype EncodedPieces = EncodedPieces [(Rank, File, Int)]

newtype EncodedRepetitions = EncodedRepetitions [Int]

data BoardState = BoardState
    { _board_position :: Position
    , _board_ply      :: Maybe Ply
    }
makeLenses ''BoardState

cNumLookBack :: Int
cNumLookBack = 7

checkmate, stalemate, draw :: Position -> Bool
checkmate pos = null (legalPlies pos) && inCheck (color pos) pos

stalemate pos = null (legalPlies pos) && not (inCheck (color pos) pos)

draw pos = insufficientMaterial pos || stalemate pos

succ_pos :: HasCallStack => BoardState -> Vector BoardState
succ_pos (BoardState pos _)
  | insufficientMaterial pos = V.empty
  | otherwise = let actions = legalPlies pos
                 in V.fromList $ map (\a -> BoardState (doPly pos a) (Just a)) actions

uniformly_choose gen options = do
    a <- uniformRM (0, V.length options - 1) gen
    return $ options V.! a

data Outcome = Win Color
             | Draw
             | Undecided
    deriving (Eq, Show)

judge :: BoardState -> Outcome
judge (BoardState pos _)
  | checkmate pos = Win (color pos)
  | stalemate pos = Draw
  | draw pos      = Draw
  | otherwise     = Undecided

reward :: Outcome -> Float
reward (Win White) = 1
reward (Win Black) = -1
reward Draw        = 0
reward Undecided   = error "the game should continue"

-- | Ascend the traversal tree till the root to get a non-empty list of cursors that
--   leads to the given cursor.
history :: Cursor BoardState -> NonEmpty (Cursor BoardState)
history c = walk (c NE.:| []) c
    where
        walk :: NonEmpty (Cursor BoardState) -> Cursor BoardState -> NonEmpty (Cursor BoardState)
        walk hist c@(Cursor z) =
            let pos = z & view (focus . node_v . board_position)
             in case parent z of
                  Nothing                     -> hist
                  Just (pz, Dict, Dict, Dict) -> let c = Cursor pz
                                                  in walk (c NE.<| hist) c

-- | Get the board for the given cursor
getPosition :: Cursor BoardState -> Position
getPosition (Cursor cur) = view (focus . node_v . board_position) cur

-- | Self-play a game with the given random generator and the number rollouts in
--   simulation step.
--
--   Returns the final cusor and full list of play steps.
play :: HasCallStack => Choose IO BoardState -> Int -> IO (Cursor BoardState, [Ply])
play choose n_rollout = do
    let cur = Cursor $ zipper $ Node (BoardState startpos Nothing) 0 0 V.empty
    runWriterT $ go cur
    where
        go :: Cursor BoardState -> WriterT [Ply] IO (Cursor BoardState)
        go cur@(Cursor z) = do

            case z & view (focus . node_v . board_ply) of
              Just ply -> tell [ply]
              Nothing  -> return ()

            next <- mcts succ_pos (liftIO . choose) (reward . judge) n_rollout cur
            case next of
              Nothing         -> return cur
              Just (sel, cur) -> go (childAt sel cur)

-- | Calculate the square after swapping the W/B plays.
rotateSquare :: (Rank, File) -> (Rank, File)
rotateSquare (r, f) = let rr = toEnum (fromEnum Rank8 - fromEnum r)
                          rf = toEnum (fromEnum FileH - fromEnum f)
                       in (rr, rf)

-- | Calculate the board pieces after swapping the W/B players.
rotateEncodedPieces :: EncodedPieces -> EncodedPieces
rotateEncodedPieces (EncodedPieces ps) = EncodedPieces (map rotate ps)
    where
        rotate (rank, file, piece) =
            let (rank', file') = rotateSquare (rank, file)
                piece' | piece < 6 = piece + 6
                       | otherwise = piece - 6
             in (rank', file', piece')

-- | Encode a step into integer
encodePly :: HasCallStack => Color -> Ply -> Maybe Int
encodePly color ply = encQueenMove <|> encKnightMove <|> encUnderPromotion
    where
        promo = plyPromotion ply
        src = plySource ply ^. rankFile
        dst = plyTarget ply ^. rankFile

        rotateIfBlack p
          | color == White = p
          | color == Black = rotateSquare p
        (src_rank, src_file) = rotateIfBlack src & bimap fromEnum fromEnum
        (tgt_rank, tgt_file) = rotateIfBlack dst & bimap fromEnum fromEnum

        vdelta = tgt_rank - src_rank
        hdelta = tgt_file - src_file
        is_horizontal = vdelta == 0
        is_vertical = hdelta == 0
        is_diagnoal = abs hdelta == abs vdelta

        encQueenMove = do
            guard (promo `elem` [Nothing, Just Queen])
            guard (is_horizontal || is_vertical || is_diagnoal)
            let distance = max vdelta hdelta - 1
                direction = case (signum vdelta, signum hdelta) of
                              (1, 0)   -> 0
                              (1, 1)   -> 1
                              (0, 1)   -> 2
                              (-1, 1)  -> 3
                              (-1, 0)  -> 4
                              (-1, -1) -> 5
                              (0, -1)  -> 6
                              (1, -1)  -> 7
                move_type = ravel [8, 7] [direction, distance]
            return $ ravel [8, 8, 73] [src_rank, src_file, move_type]

        encKnightMove = do
            direction <- case (vdelta, hdelta) of
                           (2, 1)   -> pure 0
                           (1, 2)   -> pure 1
                           (-1, 2)  -> pure 2
                           (-2, 1)  -> pure 3
                           (-2, -1) -> pure 4
                           (-1, -2) -> pure 5
                           (1, -2)  -> pure 6
                           (2, -1)  -> pure 7
                           _        -> empty
            return $ ravel [8, 8, 73] [src_rank, src_file, 56 + direction]

        encUnderPromotion = do
            guard (src_rank == 6 && tgt_rank == 7)
            guard (promo `elem` [Just Knight, Just Bishop, Just Rook])
            let direction = case hdelta of
                              -1 -> 0
                              0  -> 1
                              1  -> 2
                promo_type = case promo of
                               Just Knight -> 0
                               Just Bishop -> 1
                               Just Rook   -> 2
                move_type = ravel [3, 3] [direction, promo_type]
            return $ ravel [8, 8, 73] [src_rank, src_file, 64 + move_type]


-- | Encode the meta data
encodeBoardMeta :: Position -> [Int]
encodeBoardMeta pos =
    let ply_color = color pos
        opp_color = opponent ply_color
        turn = case color pos of
                 White -> 1
                 Black -> 0
        num_fullmove = moveNumber pos
        num_halfmove = halfMoveClock pos
        -- castling rights
        cr = castlingRights pos
        ply_kingside_castling_rights  = (ply_color, Kingside)  `elem` cr
        ply_queenside_castling_rights = (ply_color, Queenside) `elem` cr
        opp_kingside_castling_rights  = (opp_color, Kingside)  `elem` cr
        opp_queenside_castling_rights = (opp_color, Queenside) `elem` cr

     in [ turn, num_fullmove
        , fromEnum ply_kingside_castling_rights
        , fromEnum ply_queenside_castling_rights
        , fromEnum opp_kingside_castling_rights
        , fromEnum opp_queenside_castling_rights
        , num_halfmove ]

-- | Encode the pieces
encodeBoardPieces :: Position -> EncodedPieces
encodeBoardPieces pos = EncodedPieces $ catMaybes $ map read_square [minBound..maxBound]
    where
        read_square square = pieceAt pos square <&> to_ind (rank square) (file square)
        to_ind rank file (color, piece_type) =
            let offset = case color of
                           White -> 0
                           Black -> 6
             in (rank, file, fromEnum piece_type + offset)

-- | Encode the repetition status
encodeBoardRepetitions :: [Position] -> EncodedRepetitions
encodeBoardRepetitions hist@(last:_) =
    let count = length $ filter (== last) hist
        rept2 = fromEnum (count == 2)
        rept3 = fromEnum (count == 3)
     in EncodedRepetitions [rept2, rept3]

-- | Encode several information and prepare input/output for neural network's training
prepareTraining :: MonadIO m
                => Cursor BoardState -> m [(NDArray Float, NDArray Float, Float)]
prepareTraining c@(Cursor z)
  | outcome == Undecided = error "cannot get training data from an unfinished game"
  | otherwise = let stepwise = tail . inits
                    -- dropping the last step, and make replays (from step 0 to n) at each step
                    replay_each_step = stepwise $ NE.init $ history c
                    -- check the 2/3 repetition status of each replay
                    repetition_status = map checkRepetition replay_each_step
                 in liftIO $ zipWithM build replay_each_step $ stepwise repetition_status
    where
        outcome = z & view (focus . node_v) & judge

        checkRepetition :: [Cursor BoardState] -> EncodedRepetitions
        checkRepetition = encodeBoardRepetitions . map getPosition

        rotateIfBlack White = id
        rotateIfBlack Black = rotateEncodedPieces

        -- build :: [Cursor BoardState] -> [EncodedRepetitions] -> IO _
        build replay reps_enc = do
            -- encode the following
            -- * meta
            -- * pieces
            -- * 2/3 repetition
            let rev_cur@(last_cur:_) = reverse replay
                rev_pos@(last_pos:_) = map getPosition rev_cur
                color_   = color last_pos

                outcome_ = reward outcome * case color_ of
                                              White ->  1
                                              Black -> -1

                meta :: [Float]
                meta     = map fromIntegral $ encodeBoardMeta last_pos

                reps :: [[Float]]
                reps     = map (\(EncodedRepetitions a) -> map fromIntegral a) $
                           reverse $ take cNumLookBack reps_enc

                encode   = rotateIfBlack color_ . encodeBoardPieces

                enc_replay :: [EncodedPieces]
                enc_replay = map encode $ reverse $ take cNumLookBack rev_pos

                count s  = let a = fromJust $ s ^. node_v . board_ply
                               n = s ^. node_n
                            in (a, n)
                norm ac  = let (as, cs) = V.unzip ac
                               s = fromIntegral $ sum cs
                               as' = V.map (\a -> [encodePly color_ a & fromJust]) as
                               cs' = V.map (\c -> fromIntegral c / s) cs
                            in V.zip as' cs'

                actions :: Vector ([Int], Float)
                actions  = let pos = case last_cur of
                                       Cursor z -> z & view (focus . node_children)
                            in norm $ V.map count $ pos
            -- meta: (7,)
            meta  <- toNDArray [1, 1, 7] meta
            -- repetition flags: (cNumLookBack x 2, )
            reps  <- toNDArrayPadded [1, 1, cNumLookBack * 2] (concat reps)

            -- board feature: (8, 8, cNumLookBack x 12)
            board <- let cvt (EncodedPieces pl) =
                             map (\(r, f, i) -> ([fromEnum r, fromEnum f, i], 1)) pl
                      in mapM (scatter [8, 8, 12] . cvt) enc_replay

            board <- let board_len = length board
                      in case compare board_len cNumLookBack of
                           EQ -> pure board
                           GT -> error "too many boards"
                           LT -> do
                               padding <- sequence $ replicate (cNumLookBack - board_len) $
                                          ndZeros [8, 8, 12]
                               pure (padding ++ board)

            -- broadcast meta, reps to (8, 8, x)
            let b0 = head board
            meta  <- T.broadcastLikeAxis (meta, [0, 1]) (b0, [0, 1])
            reps  <- T.broadcastLikeAxis (reps, [0, 1]) (b0, [0, 1])
            -- input: (8, 8, cNumLookBack x 14 + 7), meta <+> reps <+> board
            input <- T.concat_ (-1) (board ++ [reps, meta])
            -- pi : (4672,)
            pi    <- scatter [4672] $ V.toList actions
            return (input, pi, outcome_)

        scatter :: [Int] -> [([Int], Float)] -> IO (NDArray Float)
        scatter shape idx_val = do
            let (idx, val) = unzip idx_val
                idx_vec = VS.fromList $ map fromIntegral $ concat idx
                val_vec = VS.fromList val
            val_arr <- fromVector [VS.length val_vec] val_vec
            idx_arr <- fromVector [VS.length val_vec, length (head idx)] idx_vec
                       >>= flip T.transpose [1, 0]
            T.scatter idx_arr val_arr shape

ravel :: [Int] -> [Int] -> Int
ravel shape index
  | length shape /= length index = error "mismatched length of shape and index"
  | otherwise = let dims = tail $ scanr (*) 1 shape
                in sum $ zipWith (*) dims index
                -- I may need to exploit the Ix class
                -- to calculate the linear index

toNDArray :: NumericDType a => [Int] -> [a] -> IO (NDArray a)
toNDArray shp dat = fromVector shp $ VS.fromList dat

toNDArrayPadded :: NumericDType a => [Int] -> [a] -> IO (NDArray a)
toNDArrayPadded shp dat = do
    a1 <- toNDArray [dat_len] dat
    case compare pad_len 0 of
      LT -> error "expected shape is smaller than the data"
      EQ -> reshape shp a1
      GT -> do a2 <- ndZeros [pad_len]
               concat_ 0 [a2, a1] >>= reshape shp
    where
        dat_vec = VS.fromList dat
        dat_len = VS.length dat_vec
        pad_len = product shp - dat_len
