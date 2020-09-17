from qlearningAgents import ApproximateQAgent


def test_param(vals, param, num_training, num_testing):
    """
    :param vals: A list of values to use as the parameter values
    :param param: The name of the parameter we are testing
    :return: A tuple where the first entry is a list of the max scores and the second entry
    is a list of the average scores. These scores are in sync index-wise with the :param vals
    """
    max_scores = []
    average_scores = []
    for val in vals:
        with open('test_results.txt', 'a') as f:
            print('Using value %.2f' % val, file=f)
        print('Using value %.2f' % val)
        agent = ApproximateQAgent(num_training=0, num_testing=num_testing)
        if param == 'alpha':
            agent.alpha = val
        elif param == 'gamma':
            agent.gamma = val
        elif param == 'epsilon':
            agent.epsilon = val
        elif param == 'reward_0':
            agent.reward_weights[0] = val
        elif param == 'reward_1':
            agent.reward_weights[1] = val
        elif param == 'reward_2':
            agent.reward_weights[2] = val

        agent.num_training = num_training
        agent.run_training_rounds()
        val_scores = agent.run_testing_rounds()
        max_scores.append(max(val_scores))
        avr_score = sum(val_scores) / len(val_scores)
        average_scores.append(avr_score)
        print()
    return max_scores, average_scores


def test_one_param(param_name, vals, num_training, num_testing):
    """Tests one parameter with all of the different values given"""
    with open('test_results.txt', 'a') as f:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', file=f)
        print('Testing the {} parameter with the following values: {}'.format(param_name, vals), file=f)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Testing the {} parameter with the following values: {}'.format(param_name, vals))
    max_scores, average_scores = test_param(vals, param_name, num_training, num_testing)
    with open('test_results.txt', 'a') as f:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', file=f)
        print(f"Max scores with param {param_name}: {max_scores}", file=f)
        print(f"Average scores with param {param_name}: {average_scores}", file=f)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


if __name__ == "__main__":

    alpha_vals = [0.05, 0.07, 0.09]
    gamma_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    epsilon_vals = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16]
    reward_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    parameter_names = ['alpha', 'gamma', 'epsilon', 'reward_0', 'reward_1', 'reward_2']
    parameter_values = [alpha_vals, gamma_vals, epsilon_vals, reward_weights, reward_weights, reward_weights]

    num_training_rounds = 9
    num_testing_rounds = 3

    for i in range(len(parameter_names)):
        test_one_param(parameter_names[i], parameter_values[i], num_training_rounds, num_testing_rounds)

    # q_agent = TestingApproximateQAgent(num_training=0, num_testing=10)
    # q_agent.approx_q_agent.weights = {  # Weights found during testing
    #     'bias': -190.83667758428328,
    #     'skyline_diff': -1514.7129500869028,
    #     'max_skyline_diff': -2211.239718486838,
    #     'num_holes': -8435.39859867022,
    #     'max_height': -606.511815161419,
    #     'num_rows_cleared': 147.0848355640954
    # }
    #
    # scores = q_agent.run_testing_rounds()
    # print(scores)
