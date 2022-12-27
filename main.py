import environments as env
import sketch as sh
import algorithms as alg
import math


def main():
    cliff = env.Cliff()
    labyrinth = env.Labyrinth()
    ttt_1 = env.TicTacToe(1)
    ttt_2 = env.TicTacToe(2)

    print("What would Information would you like?")
    print("1.  execute Monte Carlo with the labyrinth model (as in figure 18)")
    print("2.  average execution of Monte Carlo with the labyrinth model (as in figure 19)")
    print("3.  average execution of TD(0) compared to MC const alpha in the labyrinth model (as in figure 20)")
    print("4.  execute TD(0) with the labyrinth model (as in figure 21)")
    print("5.  average execution of different n-step TD algorithms in the labyrinth model (as in figure 23)")
    print("6.  execute SARSA with the labyrinth model (as in figure 26)")
    print("7.  average execution of SARSA compared to TD(0) in the labyrinth model (as in figure 27)")
    print("8.  average execution of Q-Learning compared to SARSA in the labyrinth model (as in figure 30)")
    print("9.  average execution of Q-Learning compared to SARSA in the cliff walking example with no epsilon decay "
          "(as in figure 32)")
    print("10. average execution of Q-Learning compared to SARSA in the cliff walking example with epsilon decay "
          "(as in figure 34)")
    print("11. average execution of Q-Learning with the TicTacToe model as player 1 (as in figure 35)")
    print("12. average execution of Q-Learning with the TicTacToe model as player 1 showing the win rate"
          " (as in figure 36)")
    print("13. average execution of Q-Learning with the TicTacToe model as player 2 showing the win rate"
          " (as in figure 37)")
    print("13. average execution of Q-Learning vs Q-Learning with the TicTacToeVS model showing the win rate"
          " (as in figure 37)")

    num = input("please input the number of the wanted result \n")
    print("\n")

    if num == "1":
        sh.execute_monte_carlo(labyrinth)
    elif num == "2":
        print("This takes some calculation time")
        sh.avg_monte_carlo(labyrinth, 1000)
    elif num == "3":
        print("This takes some calculation time")
        sh.vs2(labyrinth, 500, 1000, alg.td_0, "TD(0)", alg.monte_carlo_alpha, "MC")
    elif num == "4":
        sh.execute_td_0(labyrinth)
    elif num == "5":
        print("This takes over 1h of calculation time")
        sh.vs_multi_with_par(labyrinth, 1000, [alg.td_0, alg.td_n, alg.td_n, alg.td_n, alg.td_n, alg.td_n,
                                               alg.monte_carlo_alpha],
                                ["1-step TD", "2-step TD", "4-step TD", "6-step TD", "8-step TD", "16-step TD", "MC"],
                                "Comparing TD Algorithms", n_list=[None, 2, 4, 6, 8, 16, None])
    elif num == "6":
        sh.execute_sarsa(labyrinth)
    elif num == "7":
        print("This takes some calculation time")
        sh.vs_multi_with_par(labyrinth, 1000, [alg.sarsa, alg.td_0], ["SARSA", "TD(0)"], "SARSA compared to TD(0)",
                             alpha_list=0.1)
    elif num == "8":
        print("This takes some calculation time")
        sh.vs_multi_with_par(labyrinth, 1000, [alg.q_learning, alg.sarsa], ["Q-Learning", "SARSA"],
                             "Q-Learning compared to SARSA",
                             alpha_list=0.1)
    elif num == "9":
        print("This takes some calculation time")
        sh.vs_multi_with_par(cliff, 1000, [alg.sarsa, alg.q_learning],
                             ["SARSA", "Q-Learning"], "Cliff Walking",
                             prev_state_list=True,
                             episodes_list=1000,
                             epsilon_list=0.2, epsilon_decay_list=1)
    elif num == "10":
        print("This takes some calculation time")
        sh.vs_multi_with_par(cliff, 1000, [alg.sarsa, alg.q_learning],
                             ["SARSA", "Q-Learning"], "Cliff Walking",
                             prev_state_list=True,
                             episodes_list=1000,
                             epsilon_list=0.2)
    elif num == "11":
        print("This takes some calculation time")
        sh.avg_q_learning(ttt_1, 1000, 1000)
    elif num == "12":
        print("This takes some calculation time")
        sh.avg_q_learning_tictactoe(episodes=1000, player=1, smart=False)
    elif num == "13":
        print("This takes some calculation time")
        sh.avg_q_learning_tictactoe(episodes=1500, player=2, smart=False)
    elif num == "14":
        print("This takes over 3h of calculation time")
        sh.avg_q_learning_tictactoe(episodes=15000, smart=True)


if __name__ == '__main__':
    main()
