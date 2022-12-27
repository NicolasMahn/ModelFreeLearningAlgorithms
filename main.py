import environments as env
import sketch as sh
import algorithms as alg
import math


def main():
    cliff = env.Cliff()
    labyrinth = env.Labyrinth()
    ttt = env.TicTacToe(1)

    # sh.execute_monte_carlo(labyrinth)
    # sh.avg_monte_carlo(labyrinth, 20)

    # sh.execute_td_0(labyrinth)
    # sh.avg_td_0(labyrinth, 1000)
    # sh.vs2(labyrinth, 500, 1000, alg.td_0, "TD(0)", alg.monte_carlo_alpha, "MC")

    # sh.vs_multi_with_par(labyrinth, 1000, [alg.td_0, alg.td_n, alg.td_n, alg.td_n, alg.td_n, alg.td_n,
    #                                    alg.monte_carlo_alpha],
    #                     ["1-step TD", "2-step TD", "4-step TD", "6-step TD", "8-step TD", "16-step TD", "MC"],
    #                     "Comparing TD Algorithms", n_list=[None, 2, 4, 6, 8, 16, None])

    # sh.vs_multi_with_par(ttt, 100, [alg.q_learning, alg.q_learning, alg.q_learning],
    #                     ["alpha = 0.3", "alpha = 0.1", "alpha = 0.01"],
    #                     "Comparing epsilons for Q-Learning",
    #                     episodes_list=1000,
    #                     epsilon_list=0, epsilon_decay_list=1,
    #                     alpha_list=[0.3, 0.1, 0.01])

    # sh.avg_q_learning_tictactoe(episodes=15000, smart=True)

    #sh.vs_multi_with_par(cliff, 1000, [alg.sarsa, alg.q_learning],
    #                     ["SARSA", "Q-Learning"], "Cliff Walking",
    #                     prev_state_list=True,
    #                     episodes_list=1000,
    #                     epsilon_list=0.2, epsilon_decay_list=1)

    sh.vs_multi_with_par(cliff, 1000, [alg.sarsa, alg.q_learning],
                         ["SARSA", "Q-Learning"], "Cliff Walking",
                         prev_state_list=True,
                         episodes_list=1000,
                         epsilon_list=0.2)

    # vs_multi(cliff, 100, [quality_algorithm.sarsa, quality_algorithm.q_learning],
    #          ["SARSA", "Q-Learning"],
    #          [None, None], "Cliff Walking")

    # vs_multi(labyrinth, 1000, [monte_carlo_algorithm.monte_carlo, monte_carlo_algorithm.monte_carlo_constant_alpha],
    #         ["classical MC", "MC const alpha 0.5"],
    #         [None, 2, 4, 8, 16, 32, None], "Comparing MC Algorithms" episodes_list, gamma_list, epsilon_list, alpha_list, epsilon_decay_list)

    # vs_multi(labyrinth, 1000, [value_algorithm.td_0, monte_carlo_algorithm.monte_carlo], ["TD(0)", "MC"], "TD(0) compared to MC")

    # vs_multi_with_par(labyrinth, 1000, [quality_algorithm.q_learning, quality_algorithm.sarsa], n_list=[None,None], episodes_list=[1000,1000], gamma_list=[None,None], epsilon_list=[None,None], alpha_list=[0.1, 0.1], epsilon_decay_list=[None,None,], function_names=["Q-Learning", "SARSA"], title="Q-Learning compared to SARSA")

    # vs_multi_with_par(labyrinth, 1000, [quality_algorithm.sarsa, value_algorithm.td_0], n_list=[None,None], episodes_list=[None,None], gamma_list=[None,None], epsilon_list=[None,None], alpha_list=[0.1, 0.1], epsilon_decay_list=[None,None,], function_names=["SARSA", "TD(0)"], title="SARSA compared to TD(0)")

    # vs_multi_with_par(labyrinth, 100, functions=[quality_algorithm.sarsa, quality_algorithm.sarsa, quality_algorithm.sarsa, value_algorithm.td_0, value_algorithm.td_n, quality_algorithm.q_learning], n_list=[None,None,None,None,2,None], episodes_list=[None,None,None,None,None,None], gamma_list=[None,None,None,None,None,None], epsilon_list=[None,None,None,None,None,None], alpha_list=[0.5, 0.1, 0.01, 0.01, 0.01, 0.01], epsilon_decay_list=[None,None,None, None, None,None], function_names=["alpha 0.5", "alpha 0.1", "alpha 0.01", "TD(0)", "2-step TD", "Q"], title="SARSA alpha test")

    # sh.vs2(labyrinth, 100, 500, value_algorithm.td_0, "TD(0)", monte_carlo_algorithm.monte_carlo, "MC")
    # execute_td_0(labyrinth)
    # execute_q_learning(labyrinth)
    # avg_q_learning(labyrinth, 1000)
    # execute_monte_carlo(labyrinth)
    # avg_monte_carlo(labyrinth, 1000)
    # avg_td_0(labyrinth, 100)

    # execute_q_learning(cliff)
    # execute_sarsa(cliff)
    # avg_sarsa(cliff, 100)
    # avg_q_learning(tictactoe, 100)

    # execute_q_learning(labyrinth)

    # execute_q_learning(tictactoe_smart)
    # sh.avg_q_learning(tictactoe, 1000, episodes=1000)

    # avg_q_learning_tictactoe(number_of_iterations=1000, episodes=10000, smart=True, title="TicTacToe | Q-Learning vs Q-Learning")

    # avg_q_learning_tictactoe(player=2, number_of_iterations=1000, episodes=1000, smart=False, title="TicTacToe | Random vs Q-Learning")


if __name__ == '__main__':
    main()
