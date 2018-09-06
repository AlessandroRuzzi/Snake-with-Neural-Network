try:
    from snake_prova import Snake
    import random
    import numpy as np
    import tflearn
    import tensorflow as tf
    import math
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    from statistics import mean, median
    from collections import Counter
except:
    print('error')


class net():

    def __init__(self, initial_games=100000, test_games=40, goal_steps=10000,lr =1e-2):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr


    def initial_popolation(self):
        training_bord= []
        training_center =[]
        key = 0
        for _ in range(self.initial_games):
            if( _ %10000 == 0):
               pass
            game = Snake()

            _, prev_score,food, snake, prev_action= game.game_start(key % 4)
            key += 1

            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)

            for _ in range(self.goal_steps):

                action = int(random.randint(0, 3))
                prev_observation = self.add_action_to_observation(prev_observation, prev_action)   #i put the prev_action to the data to tell the snake in which direction it was directed

                done, score, food, snake,prev_action = game.step(action=action)

                label = np.zeros(4)
                if done:
                    break
                else:
                    food_distance = self.get_food_distance(snake,food)

                    if (food_distance < prev_food_distance or prev_score < score) and len(prev_observation) == 5:   #if is blocked i put the data in the training_board
                        label[action] += 1
                        training_bord.append([prev_observation, label])
                    elif (food_distance < prev_food_distance or prev_score < score) and len(prev_observation) == 4:  #if is blocked i put the data in the training_center
                        label[action] += 1
                        training_center.append([prev_observation, label])


                    prev_action = action/4 - 0.5
                    prev_observation = self.generate_observation(snake, food)
                    prev_score = score
        return training_bord,training_center

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_distances(self,snake):
        return (29 -snake[0][0])/ 29,snake[0][0]/29,snake[0][1]/29, (29 - snake[0][1]) /29

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        isleftblocked = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        isrightblocked = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        isfrontblocked = self.is_direction_blocked(snake, snake_direction)
        angle = self.get_angle(snake_direction, food_direction)
        food_distance = (self.get_food_distance(snake,food) / 29) - 0.5
        if isleftblocked == 0 and isrightblocked == 0 and isfrontblocked ==0 :     #if is free i put the data in the training_center
            pos = self.whereis(snake)
            #right,left,up,down = self.get_distances(snake)
            #return np.array([int(pos), round(angle, 2),round(right, 1),round(left, 1),round(up, 1),round(down, 1)])
            return np.array([pos,round(angle,2),round(food_distance,2)])

        else:                                                                     #if is blocked i put the data in the training_board
            pos = self.whereis_bord(snake)
            return np.array([int(isleftblocked), int(isrightblocked), int(isfrontblocked),pos])


    def whereis(self, snake):  #if is not blocked i tell the snake in which side of the map it was
        if  (snake[0][0] <=15 and snake[0][1] <= snake[0][0] ) or (snake[0][0]>=15 and snake[0][1] <=(29 - snake[0][0])):
            return -0.5
        elif  (snake[0][0] <=15 and (29 -snake[0][1]) <= snake[0][0] ) or (snake[0][0]>=15 and (29 -snake[0][1]) <=(29 - snake[0][0])):
            return -0.25
        if  (snake[0][1] <=15 and snake[0][0] <= snake[0][1] ) or (snake[0][1]>=15 and snake[0][0] <=(29 - snake[0][1])):
            return 0.25
        else:
            return 0.5

    def whereis_bord(self, snake): #if it is blocked i tell him in which board it was
        if snake[0][0] == 0:
            return -0.5
        elif snake[0][0] == 29:
            return -0.25
        elif snake[0][1] == 0:
            return 0.25
        elif snake[0][1] == 29:
            return 0.5
        else:
            return 0

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def normalize_vector(self, vector):
           return vector / np.linalg.norm(vector)


    def get_snake_direction_vector(self, snake):
        return np.array(snake[0] - np.array(snake[1]))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0] + np.array(direction))
        return point.tolist() in snake[:-1] or point[0] == -1 or point[1] == -1 or point[0] == 30 or point[1] == 30

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], - vector[0]])

    def net_model(self):
        g1 = tf.Graph()
        g2 = tf.Graph()

        with g1.as_default():  #first model for the board and when the snake is blocked by his body
            network = input_data(shape=[None, 5, 1], name='input')
            network = fully_connected(network, 25 , activation='relu')
            network = fully_connected(network, 4, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=self.lr, loss='categorical_crossentropy', name='predict')
            model = tflearn.DNN(network, tensorboard_dir='log')

        with g2.as_default(): #second model when the snake is free on every direction
            network_1= input_data(shape=[None, 4, 1], name='input')

            network_1= fully_connected(network_1, 128, activation='relu')
            network_1= fully_connected(network_1, 4, activation='softmax')
            network_1= regression(network_1, optimizer='adam', learning_rate=self.lr, loss='categorical_crossentropy', name='predict')
            model_1= tflearn.DNN(network_1, tensorboard_dir='log')

        return model,model_1

    def train_net(self,training_bord, training_center, model_1,model):

        X = np.array([i[0] for i in training_bord]).reshape((-1, 5, 1))

        Y = np.array([i[1] for i in training_bord]).reshape(-1, 4)

        X_1 = np.array([i[0] for i in training_center]).reshape((-1, 4, 1))



        Y_1 = np.array([i[1] for i in training_center]).reshape(-1, 4)

        model_1.fit(X_1, Y_1, n_epoch=1, show_metric=True,shuffle=True)
        print('***',model_1)
        model.fit(X, Y,batch_size= 48, n_epoch=3, show_metric=True, shuffle=True)

        return model,model_1

    def test_neural_net(self, model,model_1):
        steps_arr = []
        score_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = Snake()
            _,score, food, snake,prev_action = game.game_start(1)
            prev_observation = self.generate_observation(snake=snake,food = food)


            for _ in range(self.goal_steps):

                prev_observation = self.add_action_to_observation(prev_observation, prev_action)

                if steps == 0:
                    action = random.randint(0, 3)
                else:
                    if len(prev_observation) == 4:
                       action = np.argmax(model_1.predict(prev_observation.reshape(-1, 4, 1)))
                       print('***', prev_observation)
                       print(model_1.predict(prev_observation.reshape(-1, 4, 1)))
                       print(action)
                    else:
                        action = np.argmax(model.predict(prev_observation.reshape(-1, 5, 1)))
                        print('***', prev_observation)
                        print(model.predict(prev_observation.reshape(-1, 5, 1)))
                        print(action)
                done, score, food, snake,prev_action = game.play(action=action)

                prev_action = action/4 - 0.5

                game_memory.append([prev_observation, action])

                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake=snake,food=food)
                    steps += 1
            steps_arr.append(steps)
            score_arr.append(score)
        print('Avarage steps: ', mean(steps_arr))
        print(Counter(steps_arr))
        print('Avarage score: ',mean(score_arr))
        print(Counter(score_arr))

    def test(self):

        training_bord,training_center = self.initial_popolation()

        model, model_1 = self.net_model()

        model,model_1 = self.train_net(training_bord,training_center,model_1,model)

        self.test_neural_net(model,model_1)


if __name__ == '__main__':
    neural = net()
    neural.test()
