import math
from Board import Board
from ChessPiece import *
from functools import wraps
from Logger import Logger, BoardRepr
from logCSV import log_features_to_csv
from ml import evaluate_board  # Import the evaluate_board function from ML.py
import random
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


logger = Logger()


def log_tree(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        board: Board = args[0]
        if board.log:
            depth = args[1]
            write_to_file(board, depth)
        return func(*args, **kwargs)
    return wrapper


def write_to_file(board: Board, current_depth):
    global logger
    if board.depth == current_depth:
        logger.clear()
    board_repr = BoardRepr(board.unicode_array_repr(), current_depth, board.evaluate())
    logger.append(board_repr)

def is_capturing_move(board, move):
    return isinstance(board[move[0]][move[1]], ChessPiece)

def is_king_threat(board, move):
    return board.king_is_threatened(board.get_player_color(), move)

@log_tree
def minimax(board, depth, alpha, beta, max_player, save_move, data):

    if depth == 0 or board.is_terminal():
        data[1] = board.evaluate()
        return data

    if max_player:
        max_eval = -math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                    piece = board[i][j]
                    moves = piece.filter_moves(piece.get_moves(board), board)
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)
                    #if board.number_of_turn >= 10:
                     #   moves.sort(key = lambda sorted_move: is_king_threat(board, sorted_move), reverse = True)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, False, False, data)[1]
                        if save_move:
                            if evaluation >= max_eval:
                                if evaluation > data[1]:
                                    data.clear()
                                    data[1] = evaluation
                                    data[0] = [piece, move, evaluation]
                                elif evaluation == data[1]:
                                    data[0].append([piece, move, evaluation])
                        board.unmake_move(piece)
                        max_eval = max(max_eval, evaluation)
                        alpha = max(alpha, evaluation)
                        if beta <= alpha:
                            break
        return data
    else:
        min_eval = math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color == board.get_player_color():
                    piece = board[i][j]
                    moves = piece.get_moves(board)
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)
                    if board.number_of_turn >= 10:
                        moves.sort(key = lambda sorted_move: is_king_threat(board, sorted_move), reverse = True)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, True, False, data)[1]
                        board.unmake_move(piece)
                        min_eval = min(min_eval, evaluation)
                        beta = min(beta, evaluation)
                        if beta <= alpha:
                            break
        return data


def progressive_deepening(board, max_depth):
    best_evaluation = -math.inf
    best_move_data = []
    for depth in range(1, max_depth + 1):
        current_evaluation = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])[1]
        if current_evaluation > best_evaluation:
            best_evaluation = current_evaluation
            best_move_data = minimax(board, depth, -math.inf, math.inf, True, True, [[], 0])
    return best_move_data, best_evaluation

def get_ai_move(board):
    # Using progressive deepening with a max depth
    moves_data, evaluation = progressive_deepening(board, board.depth)
    if moves_data and len(moves_data[0]) > 0:
        best_score = max(moves_data[0], key=lambda x: x[2])[2]
        piece_and_move = random.choice([move for move in moves_data[0] if move[2] == best_score])
        piece, move_coords = piece_and_move[0], piece_and_move[1]
        board.make_move(piece, move_coords[0], move_coords[1])
        board.number_of_turn += 1
        
        features = extract_features(board)
        evaluation_data = {'evaluation': evaluation}
        features[0].update(evaluation_data)
        log_features_to_csv('data.csv', features)
        return True
    return False


def get_random_move(board):
    pieces = []
    moves = []
    for i in range(8):
        for j in range(8):
            if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                pieces.append(board[i][j])
    for piece in pieces[:]:
        piece_moves = piece.filter_moves(piece.get_moves(board), board)
        if len(piece_moves) == 0:
            pieces.remove(piece)
        else:
            moves.append(piece_moves)
    if len(pieces) == 0:
        return
    piece = random.choice(pieces)
    move = random.choice(moves[pieces.index(piece)])
    if isinstance(piece, ChessPiece) and len(move) > 0:
        board.make_move(piece, move[0], move[1])
        board.number_of_turn += 1


def minimax_lvl1(board, depth, alpha, beta, max_player, save_move, data):

    if depth == 0 or board.is_terminal():
        data[1] = board.evaluate()
        return data

    if max_player:
        max_eval = -math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                    piece = board[i][j]
                    moves = piece.filter_moves(piece.get_moves(board), board)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, False, False, data)[1]
                        if save_move:
                            if evaluation >= max_eval:
                                if evaluation > data[1]:
                                    data.clear()
                                    data[1] = evaluation
                                    data[0] = [piece, move, evaluation]
                                elif evaluation == data[1]:
                                    data[0].append([piece, move, evaluation])
                        board.unmake_move(piece)
                        max_eval = max(max_eval, evaluation)
                        alpha = max(alpha, evaluation)
                        if beta <= alpha:
                            break
        return data
    else:
        min_eval = math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color == board.get_player_color():
                    piece = board[i][j]
                    moves = piece.get_moves(board)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = minimax(board, depth - 1, alpha, beta, True, False, data)[1]
                        board.unmake_move(piece)
                        min_eval = min(min_eval, evaluation)
                        beta = min(beta, evaluation)
                        if beta <= alpha:
                            break
        return data


def get_ai_move_lvl1(board):
    moves = minimax_lvl1(board, board.depth, -math.inf, math.inf, True, True, [[], 0])
    if board.log:
        logger.write()
    # moves = [[pawn, move, move_score], [..], [..],[..], total_score]
    if len(moves[0]) == 0:
        return False
    best_score = max(moves[0], key=lambda x: x[2])[2]
    piece_and_move = random.choice([move for move in moves[0] if move[2] == best_score])
    piece = piece_and_move[0]
    move = piece_and_move[1]
    if isinstance(piece, ChessPiece) and len(move) > 0 and isinstance(move, tuple):
        board.make_move(piece, move[0], move[1])
    return True

def extract_features(board):
    features = []
    features.append({
        'game_phase': determine_game_phase(board),
        'material_balance': calculate_material_balance(board)
    })
    return features

def determine_game_phase(board):
    total_moves = board.number_of_turn
    if total_moves < 10:
        return 'opening'
    elif total_moves < 20:
        return 'midgame'
    else:
        return 'endgame'

def calculate_material_balance(board):
    white_material = 0
    black_material = 0
    
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if isinstance(piece, ChessPiece):
                if piece.color == 'white':
                    white_material += piece.get_score(board)
                elif piece.color == 'black':
                    black_material += piece.get_score(board)
    
    return white_material - black_material
    

# AI moves with ML
def evaluate_ML(board, depth, alpha, beta, max_player, save_move, data):

    if depth == 0 or board.is_terminal():
        features = extract_features(board)
        feature_dict = features[0]['material_balance']
        board_score = evaluate_board(feature_dict)
        data[1] = board_score
        return data

    if max_player:
        max_eval = -math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color != board.get_player_color():
                    piece = board[i][j]
                    moves = piece.filter_moves(piece.get_moves(board), board)
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)
                    #if board.number_of_turn >= 10:
                     #   moves.sort(key = lambda sorted_move: is_king_threat(board, sorted_move), reverse = True)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = evaluate_ML(board, depth - 1, alpha, beta, False, False, data)[1]
                        if save_move:
                            if evaluation >= max_eval:
                                if evaluation > data[1]:
                                    data.clear()
                                    data[1] = evaluation
                                    data[0] = [piece, move, evaluation]
                                elif evaluation == data[1]:
                                    data[0].append([piece, move, evaluation])
                        board.unmake_move(piece)
                        max_eval = max(max_eval, evaluation)
                        alpha = max(alpha, evaluation)
                        if beta <= alpha:
                            break
        return data
    else:
        min_eval = math.inf
        for i in range(8):
            for j in range(8):
                if isinstance(board[i][j], ChessPiece) and board[i][j].color == board.get_player_color():
                    piece = board[i][j]
                    moves = piece.get_moves(board)
                    moves.sort(key=lambda sorted_move: is_capturing_move(board, sorted_move), reverse=True)
                    if board.number_of_turn >= 10:
                        moves.sort(key = lambda sorted_move: is_king_threat(board, sorted_move), reverse = True)
                    for move in moves:
                        board.make_move(piece, move[0], move[1], keep_history=True)
                        evaluation = evaluate_ML(board, depth - 1, alpha, beta, True, False, data)[1]
                        board.unmake_move(piece)
                        min_eval = min(min_eval, evaluation)
                        beta = min(beta, evaluation)
                        if beta <= alpha:
                            break
        return data


def progressive_deepening_ML(board, max_depth):
    best_evaluation = -math.inf
    best_move_data = []
    for depth in range(1, max_depth + 1):
        current_evaluation = evaluate_ML(board, depth, -math.inf, math.inf, True, True, [[], 0])[1]
        if current_evaluation > best_evaluation:
            best_evaluation = current_evaluation
            best_move_data = evaluate_ML(board, depth, -math.inf, math.inf, True, True, [[], 0])
    return best_move_data

def using_ML_move(board):
    # print("ML move")
    moves_data = progressive_deepening_ML(board, board.depth)
    if moves_data and len(moves_data[0]) > 0:
        best_score = max(moves_data[0], key=lambda x: x[2])[2]
        piece_and_move = random.choice([move for move in moves_data[0] if move[2] == best_score])
        piece, move_coords = piece_and_move[0], piece_and_move[1]
        board.make_move(piece, move_coords[0], move_coords[1])
        board.number_of_turn += 1
        return True
    return False