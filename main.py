
from config import initial

if __name__ == '__main__':
    print("Welcome to modified Chess!")
    print("Please select your game mode:")
    print("1. Player vs Bot")
    print("2. Bot vs Bot")
    mode = int(input("Enter your choice: "))
    initial.setMode(mode)
    if mode == '1':
        #game_mode = 0
        print("Please select the difficulty level: ")
        print("1. Easy (Random bot)")
        print("2. Medium (Minimax bot with depth = 1)")
        print("3. Hard (Minimax bot with depth = 3)")
        print("4. Very Hard (Modified Minimax bot with depth = 1)")
        print("5. Impossible (Modified Minimax bot with depth = 3)")
        print("6. Using ML")
        difficulty = int(input("Enter your choice: "))
        initial.setDifficulty(difficulty)
        if difficulty == 3 or difficulty == 5 or difficulty == 6:
            initDepth = 3
        else:
            initDepth = 1
    else:
        print("The selected bot will be evaluated with a stupid bot, select your fighter: ")
        print("1. Easy (Random bot)")
        print("2. Medium (Minimax bot with depth = 1)")
        print("3. Hard (Minimax bot with depth = 3)")
        print("4. Very Hard (Modified Minimax bot with depth = 1)")
        print("5. Impossible (Modified Minimax bot with depth = 3)")
        print("6. Using ML")
        difficulty = int(input("Enter your choice: "))
        initial.setDifficulty(difficulty)
        if difficulty == 3 or difficulty == 5:
            initDepth = 3
        else:
            initDepth = 1
    with open('initial.txt', 'w') as f:
        f.write(f'{mode}\n')
        f.write(f'{difficulty}\n')
    keep_playing = True

    import graphics
    from Board import *
    board = Board(game_mode=0, ai=True, depth=initDepth, log=True)  # game_mode == 0: whites down / 1: blacks down

    # main.py
    

    while keep_playing:
        graphics.initialize()
        board.place_pieces()
        graphics.draw_background(board)
        keep_playing = graphics.start(board)
        