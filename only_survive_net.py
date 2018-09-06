try:
    from snake_prova import Snake
    import random
    import numpy as np
    import tflearn
    import math
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    from statistics import mean, median
    from collections import Counter
except:
    print('error')

class net():

    def __init__(self,initial_games = 100000, test_games= 40, goal_steps = 10000,lr = 1e-2,filename = 'snake_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games =test_games
        self.goal_steps =goal_steps
        self.lr = lr
        self.filename =filename

    def initial_popolation(self):  ##take data to train the snake, only data when the snake survive and noly if it was at the board of the map
        training_data= []
        key= 0
        for _ in range(self.initial_games):

            game = Snake()

            _,prev_score,food,snake,prev_action= game.game_start(key%4)
            key+=1

            prev_observation = self.generate_observation(snake)


            for _ in range(self.goal_steps):

                action = int(random.randint(0,3))
                prev_observation = self.add_action_to_observation(prev_observation, prev_action)
                done,score,_,snake,prev_action= game.step(action = action)

                label = np.zeros(4)
                if done:
                    break
                elif prev_observation[1] == 0 and prev_observation[2] == 0 and prev_observation[3]== 0 :
                    prev_action = action/4 - 0.5
                    prev_observation = self.generate_observation(snake)


                else:
                    label[action] +=1
                    training_data.append([prev_observation, label])
                    prev_action =action/4 - 0.5
                    prev_observation= self.generate_observation(snake)

        return training_data

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def generate_observation(self,snake):
        snake_direction = self.get_snake_direction_vector(snake)
        isleftblocked = self.is_direction_blocked(snake,self.turn_vector_to_the_left(snake_direction))
        isrightblocked = self.is_direction_blocked(snake,self.turn_vector_to_the_right(snake_direction))
        isfrontblocked = self.is_direction_blocked(snake,snake_direction)
        pos = self.whereis(snake)

        return np.array([int(isleftblocked),int(isrightblocked),int(isfrontblocked),int(pos)])

    def whereis(self, snake): ##this tell to the snake in which side of the map it was
        if snake[0][0] == 0:
            return 0
        elif snake[0][0] == 29:
            return 1
        elif snake[0][1] == 0:
            return 2
        elif snake[0][1] == 29:
            return 3
        else:
            return 4


    def get_snake_direction_vector(self,snake):
        return np.array(snake[0] - np.array(snake[1]))

    def is_direction_blocked(self, snake, direction): ## this tell the snake if it was at the board of the map
        point = np.array(snake[0] + np.array(direction))
        return point.tolist() in snake[:-1] or point[0] == -1 or point[1] == -1 or point[0] == 30 or point[1] == 30

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], - vector[0]])

    def net_model(self):

        network = input_data(shape=[None, 5, 1], name='input')

        network = fully_connected(network, 25, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network,4, activation='linear')

        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_net(self,training_data,model):

        X = np.array([i[0] for i in training_data]).reshape((-1,5,1))

        Y = np.array([i[1] for i in training_data]).reshape(-1,4)

        model.fit(X,Y,batch_size = 48,n_epoch = 3,show_metric=True,shuffle = True)

        model.save(self.filename)

        return model

    def test_neural_net(self,model):
        steps_arr = []

        for _  in range (self.test_games):
            steps = 0
            game_memory = []
            game= Snake()
            _,_,_,snake,prev_action= game.game_start(1)
            prev_observation = self.generate_observation(snake= snake)

            for _ in range(self.goal_steps):

                prev_observation = self.add_action_to_observation(prev_observation, prev_action)
                if steps == 0 or (prev_observation[1] == 0 and prev_observation[2] == 0 and prev_observation[3]== 0):
                    action = random.randint(0,3)
                else:
                    action = np.argmax(model.predict(prev_observation.reshape(-1,5,1)))
                    print('***', prev_observation)
                    print(model.predict(prev_observation.reshape(-1, 5, 1)))
                    print(action)
                done,_,_,snake,prev_action = game.play(action=action)

                prev_action = action/4 - 0.5

                game_memory.append([prev_observation,action])

                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake=snake)
                    steps +=1
            steps_arr.append(steps)
        print('Avarage steps: ',mean(steps_arr))
        print(Counter(steps_arr))

    def test(self):

        training_data = self.initial_popolation()

        model = self.net_model()

        model = self.train_net(training_data,model)

        self.test_neural_net(model)


if __name__ == '__main__':
    neural = net()
    neural.test()
