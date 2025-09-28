import copy
import torch
import sys
from os import path
import math
import random
sys.path.append("..")
from tournament.runner import *
from tournament import utils
from player import Team, DummyTeam
from jurgen_agent.player import extract_featuresV2
from jurgen_agent.player import Team as JurgenTeam
from geoffrey_agent.player import Team as GeoffreyTeam
from yann_agent.player import Team as YannTeam
from yoshua_agent.player import Team as YoshuaTeam
from torch.distributions import Bernoulli

random.seed()

n_epochs = 400
#n_iterations = 300
batch_size = 512
T = 5

#Staterecorder
#AIRunner
#teamRunner
#Match
def train():
    learn_team = Team()
    train_team = JurgenTeam()
    train_team.team = 0
    learn_team.team = 0
    learn_team_runner = TeamRunner(learn_team)
    train_team_runner = TeamRunner(train_team)
    possible_ai = [JurgenTeam, DummyTeam, YannTeam, GeoffreyTeam, YoshuaTeam]
    match = Match()
    trainer_model = train_team.model

    model = learn_team.k1_net

    model.train()

    acc_loss = torch.nn.SmoothL1Loss()
    steer_loss = torch.nn.BCEWithLogitsLoss()
    brake_loss = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(learn_team.k1_net.parameters(), lr=1e-3)
    big_feature_array = []
    big_train_actions_array = []
    
    for epoch in range(n_epochs):
        # Roll out the policy, compute the Expectation
        #trajectories = rollout_many([Actor(action_net)]*n_trajectories, n_steps=600)
        recorder = None
        recorder = recorder & utils.StateRecorder("state.pkl") & utils.VideoRecorder("videos/" + str(epoch) + "video.mp4")
        #recorder = recorder & utils.VideoRecorder(str(epoch) + "video.mp4")
        ai_runner = TeamRunner(random.choice(possible_ai)())
        if (epoch == 0):
            results = match.run(train_team_runner, ai_runner, num_player=2, record_fn=recorder)
        else:
            #vel_factor = 15
            spawn_window = [100, 100]
            results = match.run(learn_team_runner, ai_runner, num_player=2, record_fn=recorder, max_score=1, \
                initial_ball_location=[(random.random() - 0.5) * spawn_window[0], (random.random() - 0.5) * spawn_window[1]])
                #initial_ball_velocity=[2 * (random.random() - 0.5) * vel_factor, 2 * (random.random() - 0.5) * vel_factor])
        print("EPOCH " + str(epoch) + ": " + str(results))
        feature_array = []
        actions_array = []
        states_array = []
        reward_array = []
        train_actions_array = []
        fnum = 0
        for frame in utils.load_recording("state.pkl"):
            fnum += 1
            #if fnum < 500:
                #continue
            player_state = frame["team1_state"]
            opponent_state = frame["team2_state"]
            soccer_state = frame["soccer_state"]
            states_array.append((soccer_state, player_state))
            actions = frame["actions"][0]
            train_actions = train_team.act(player_state, opponent_state, soccer_state)[0]
            #print(actions)
            features = extract_featuresV2(player_state[0], soccer_state, opponent_state, learn_team.team)
            feature_array.append(features)
            big_feature_array.append(features)

            acceleration = train_actions["acceleration"]
            steer = 0 if train_actions["steer"] < 0 else 1
            brake = train_actions["brake"]
            train_action = [acceleration, steer, brake]
            train_actions_array.append(train_action)
            big_train_actions_array.append(train_action)


            #acceleration = actions["acceleration"]
            #steer = 0 if actions["steer"] < 0 else 1
            #brake = actions["brake"]
            #action = [acceleration, steer, brake]
            #actions_array.append(action)


        #for i in range(len(states_array)):
            #future_idx = min(i + T, len(states_array) - 1)
            #reward_array.append([reward_function(states_array[i], states_array[future_idx],  1, i)])

        #rewards = torch.as_tensor(reward_array, dtype=torch.float32).to(device)

        train_actions = torch.as_tensor(big_train_actions_array, dtype=torch.float32)
        features = torch.stack(big_feature_array)

        #rewards = (rewards - rewards.mean()) / rewards.std()

        avg_loss = []
        random_indices = list(range(len(train_actions)))
        random.shuffle(random_indices)

        
        for i in range(math.ceil(len(big_feature_array) / batch_size)):
            #batch_ids = torch.randint(0, len(train_actions), (batch_size if batch_size < len(train_actions) else len(train_actions),), device=device)
            start = batch_size * i
            end = start + batch_size if start + batch_size < len(big_feature_array) else len(big_feature_array)
            batch_ids = torch.as_tensor(random_indices[start:end], dtype=torch.long)
            #batch_rewards = rewards[batch_ids]
            batch_actions = train_actions[batch_ids]
            batch_features = features[batch_ids]

            batch_features = torch.nan_to_num(batch_features)

            output = model(batch_features)
            #acc_pi = Bernoulli(logits=output[:,0])
            #steer_pi = Bernoulli(logits=output[:,1])
            #brake_pi = Bernoulli(logits=output[:,2])

            #expected_log_return_acc = (acc_pi.log_prob(batch_actions[:, 0])*batch_rewards).mean()
            #expected_log_return_steer = (steer_pi.log_prob(batch_actions[:, 1])*batch_rewards).mean()
            #expected_log_return_brake = (acc_pi.log_prob(batch_actions[:, 2])*batch_rewards).mean()

            acc_loss_val = acc_loss(output[:, 0], batch_actions[:, 0])
            steer_loss_val = steer_loss(output[:, 1], batch_actions[:, 1])
            brake_loss_val = brake_loss(output[:, 2], batch_actions[:, 2])
            total_loss = acc_loss_val + steer_loss_val + brake_loss_val
            #negative_total_loss = expected_log_return_brake + expected_log_return_steer + expected_log_return_acc

            optim.zero_grad()
            total_loss.backward()
            #(-negative_total_loss).backward()
            optim.step()

            avg_loss.append(float(total_loss))
        print(sum(avg_loss)/len(avg_loss))
        torch.jit.save(torch.jit.script(model), path.join(path.dirname(path.abspath(__file__)), 'my_agent.pt'))

        # Compute all the reqired quantities to update the policy
def reward_function(now_state, future_state, opposing_team, frame):
    now_soccer_state = now_state[0]
    future_soccer_state = future_state[0]
    now_kart_state = now_state[1]
    future_kart_state = future_state[1]

    goal_line_center_x = (now_soccer_state["goal_line"][opposing_team][0][0] + now_soccer_state["goal_line"][opposing_team][1][0]) / 2
    goal_location2d = (goal_line_center_x, now_soccer_state["goal_line"][opposing_team][0][2])
    field_size = abs(goal_location2d[1] * 2)

    ball_location_now = now_soccer_state["ball"]["location"]
    ball_location_future = future_soccer_state["ball"]["location"]

    kart_location_now = now_kart_state["kart"]["location"]
    kart_location_future = future_kart_state["kart"]["location"]

    a = kart_location_now[0] - kart_location_future[0]
    b = kart_location_now[1] - kart_location_future[1]

    kart_displacement = math.sqrt(a**2 + b**2)

    a = goal_location2d[0] - ball_location_now[0]
    b = goal_location2d[1] - ball_location_now[2]

    ball_goal_dist_now = math.sqrt(a**2 + b**2)

    a = goal_location2d[0] - ball_location_future[0]
    b = goal_location2d[1] - ball_location_future[2]

    ball_goal_dist_future = math.sqrt(a**2 + b**2)

    a = ball_location_now[0] - kart_location_now[0]
    b = ball_location_now[2] - kart_location_now[2]

    kart_ball_dist_now = math.sqrt(a**2 + b**2)

    a = ball_location_future[0] - kart_location_future[0]
    b = ball_location_future[2] - kart_location_future[2]

    kart_ball_dist_future = math.sqrt(a**2 + b**2)

    distance_diff = -(ball_goal_dist_future - ball_goal_dist_now)
    kart_ball_dist_diff = -(kart_ball_dist_future - kart_ball_dist_now)

    score_now = now_soccer_state["score"][opposing_team]
    score_future = future_soccer_state["score"][opposing_team]

    score_contrib = field_size * 2 if score_future > score_now else 0
    ball_goal_distance_contrib = distance_diff
    kart_ball_contrib = kart_ball_dist_diff
    kart_displacement_contrib = kart_displacement * 0.1

    result = score_contrib + ball_goal_distance_contrib + kart_ball_contrib + kart_displacement_contrib
    return result

if __name__ == '__main__':
    train()
