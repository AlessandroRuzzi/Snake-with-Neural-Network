try:
    import pygame
    import sys
    import random
    import time
except:
    print('error')
#some_constant to easy change the game

BOARD_SIZE = 30           #size of the board, in block
BLOCK_SIZE =20            #size of 1 blovk,pixel
HEAD_COLOR =(0,0,255)     #dark blue-> use this link if you want to change the color-> http://www.discoveryplayground.com/computer-programming-for-kids/rgb-colors/
BODYCOLOR = (30,144,255)  #light blue
FOOD_COLOR = (200,0,0)    #dark red
GAME_SPEED = 20           #game speed -> the bigger the faster

class Snake():
    def __init__(self):
        self.head = [int(BOARD_SIZE/4),int(BOARD_SIZE/4)]
        self.body = [[self.head[0],self.head[1]],[self.head[0]-1,self.head[1]],[self.head[0]-2,self.head[1]]]
        self.direction = 'RIGHT'
        self.food = [random.randrange(1, BOARD_SIZE), random.randrange(1, BOARD_SIZE)]
        self.is_food_on_screen = True
        self.score = 0
        self.head_value = 1
        self.done = False


    def change_direction(self,dir):
        if dir == 'RIGHT' and not self.direction == 'LEFT':
            self.direction = 'RIGHT'

        if dir == 'LEFT' and not self.direction == 'RIGHT':
            self.direction = 'LEFT'

        if dir == 'UP' and not self.direction == 'DOWN':
            self.direction = 'UP'

        if dir == 'DOWN' and not self.direction == 'UP':
            self.direction = 'DOWN'

    def move_snake(self,foodpos):
        if self.direction == 'RIGHT':
            self.head[0] += 1           #move the snake to the position and remove the tail if the snake does not eat food

        if self.direction == 'LEFT':
            self.head[0] -= 1

        if self.direction == 'UP':
            self.head[1] -= 1

        if self.direction == 'DOWN':
            self.head[1] += 1

        self.body.insert(0,list(self.head))
        if self.head == foodpos:
            return 1
        else:
            self.body.pop()
            return 0

    def Collision(self):
        #check if the head collides with the walls
        if self.head[0] >BOARD_SIZE-1 or self.head[0] < 0 :
            return 1
        elif self.head[1] >BOARD_SIZE-1 or self.head[1]<0:
            return 1
        #check if the head collides with the body
        for body in self.body[1:]:
            if self.head == body:
                return 1
        return 0

    def get_body(self):
        return self.body

    def spawn_elem(self):
        if self.is_food_on_screen == False:
            self.food= [random.randrange(1,BOARD_SIZE),random.randrange(1,BOARD_SIZE)]
            self.is_food_on_screen = True
        return self.food

    def set_food_on_screen(self,bool):
        self.is_food_on_screen = bool


    def game_start(self,key):               ##the key make the snake spawn in different places of the map
        if key == 0:
            self.head = [int((BOARD_SIZE / 4) * 3), int(BOARD_SIZE / 4)]
            self.body = [[self.head[0], self.head[1]], [ self.head[0] - 1,  self.head[1]],
                          [ self.head[0] - 2,  self.head[1]]]
            self.direction = 'DOWN'                                    
        if key == 1:
            self.head = [int(BOARD_SIZE / 4), int(BOARD_SIZE / 4)]
            self.body = [[ self.head[0], self.head[1]], [ self.head[0] - 1,  self.head[1]],
                          [ self.head[0] - 2, self.head[1]]]
            self.direction = 'RIGHT'
        if key == 2:
            self.head = [int((BOARD_SIZE / 4) * 3), int((BOARD_SIZE / 4) * 3)]
            self.body = [[ self.head[0],  self.head[1]], [ self.head[0] - 1, self.head[1]],
                          [ self.head[0] - 2, self.head[1]]]
            self.direction = 'UP'
        if key == 3:
            self.head = [int(BOARD_SIZE / 4), int((BOARD_SIZE / 4)) * 3]
            self.body = [[ self.head[0],  self.head[1]], [ self.head[0] - 1,  self.head[1]],
                          [ self.head[0] - 2, self.head[1]]]
            self.direction = 'UP'
        self.window = pygame.display.set_mode((BOARD_SIZE * BLOCK_SIZE, BOARD_SIZE * BLOCK_SIZE))
        pygame.display.set_caption('Snake Game')
        self.fps = pygame.time.Clock()
        return self.generate_observation()
        ''''for i in range(3):
          pygame.display.set_caption('SNAKE GAME | Game Starts in ' + str(3-i) +' second(s)...')
          pygame.time.wait(1000)'''''


    def generate_observation(self):
        return self.done,self.score,self.food,self.body


    def game_over(self):
       #pygame.display.set_caption('SNAKE GAME | Score: ' + str(self.score) + ' | GAME OVER -> Press any key to leave ')
      # while True:
          #event = pygame.event.wait()
          #if event.type == pygame.KEYDOWN:
             #break
       pygame.quit()
       sys.exit()

    def play(self,action= 0,color = True ):

            '''for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.change_direction('RIGHT')

                    if event.key == pygame.K_LEFT:
                        self.change_direction('LEFT')

                    if event.key == pygame.K_UP:
                        self.change_direction('UP')

                    if event.key == pygame.K_DOWN:
                        self.change_direction('DOWN')'''
            if int(action) == 0:
                self.change_direction('RIGHT')

            if int(action) == 1:
                self.change_direction('LEFT')

            if int(action == 2):
                self.change_direction('UP')

            if int(action) == 3:
                self.change_direction('DOWN')

            foodpos = self.spawn_elem()
            if self.move_snake(foodpos) == 1:
                self.score += 1
                self.set_food_on_screen(False)


            if color:
                self.window.fill(pygame.Color(0,0,0))                    ##to set the background color to black or white
            else:
                self.window.fill(pygame.Color(225, 225, 225))

            self.head_value = 1 
            
            #draw snake
            for pos in self.get_body():
                if self.head_value == 1:
                    pygame.draw.rect(self.window, HEAD_COLOR, pygame.Rect(pos[0] * BLOCK_SIZE, pos[1] * BLOCK_SIZE,
                                                                     BLOCK_SIZE, BLOCK_SIZE))
                    self.head_value = 0

                else:
                    pygame.draw.rect(self.window, BODYCOLOR, pygame.Rect(pos[0] * BLOCK_SIZE, pos[1] * BLOCK_SIZE,
                                                                    BLOCK_SIZE, BLOCK_SIZE))
            # draw food

            pygame.draw.rect(self.window, FOOD_COLOR, pygame.Rect(foodpos[0] * BLOCK_SIZE, foodpos[1] * BLOCK_SIZE,
                                                             BLOCK_SIZE, BLOCK_SIZE))

            if self.Collision() == 1:
                self.done =True
                return self.generate_observation()
                self.game_over()

            pygame.display.set_caption('SNAKE GAME | Speed: ' + str(GAME_SPEED) + ' | Score: ' + str(self.score))
            pygame.display.flip()
            self.fps.tick(GAME_SPEED)

            return self.generate_observation()
    def step(self,action = 0):                             
        if int(action) == 0:
            self.change_direction('RIGHT')
                                                             ##to send data to the neural net whitout visualise the game
        if int(action) == 1:
            self.change_direction('LEFT')

        if int(action == 2):
            self.change_direction('UP')

        if int(action) == 3:
            self.change_direction('DOWN')

        foodpos = self.spawn_elem()
        if self.move_snake(foodpos) == 1:
            self.score += 1
            self.set_food_on_screen(False)

        if self.Collision() == 1:
            self.done = True
            return self.generate_observation()
            self.game_over()

        return self.generate_observation()


if __name__ == '__main__':
    game = Snake()
    game.game_start()
    for _ in range(20):
        game.play(action = (random.randint(0,3)))




