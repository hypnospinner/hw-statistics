import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

# task 1

# brown - yellow - orange - green


def task1():
    print("TASK 1")

    observed = np.array([42, 53, 64, 65])
    expected = np.array([observed.sum() / len(observed)
                         for i in range(len(observed))])
    impericProbs = np.array([v / observed.sum()
                             for i, v in enumerate(observed)])
    R_i = np.array([((expected[i] - observed[i]) ** 2) / expected[i]
                    for i in range(len(observed))])
    R_cr = stats.chi2.ppf(q=0.95, df=len(observed)-1)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"Imperic Probabilities:\n{impericProbs}\n")

    print(f"R_i:\n{R_i}\n")

    print(f"R: {R_i.sum()}\n")

    print(f"R_cr: {R_cr}\n")

    print("Accepted at 0.05. Machine has no preferences")

    # output graph into file

    ox = ["brown", "yellow", "orange", "green"]
    plt.plot(ox, observed, 'r')
    plt.plot(ox, expected, 'b')
    plt.show()

# task 2


def task2():
    print("TASK 2")

    observed = np.array([0, 10, 10, 10, 15, 15])
    expected = np.array([observed.sum() / len(observed)
                         for i in range(len(observed))])
    R_i = np.array([((expected[i] - observed[i]) ** 2) / expected[i]
                    for i in range(len(observed))])
    R_cr = stats.chi2.ppf(q=0.95, df=len(observed)-1)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"R_i:\n{R_i}\n")

    print(f"R: {R_i.sum()}\n")

    print(f"R_cr: {R_cr}\n")

    print(f"Rejected at 0.05. Cube is wrong")

    ox = [i for i in range(1, len(observed)+1)]
    plt.plot(ox, observed, 'r')
    plt.plot(ox, expected, 'b')
    plt.show()


def task3():
    print("TASK 3")


    observed = np.array([8, 22, 14, 9, 12])
    expected = np.array([observed.sum() / len(observed)
                         for i in range(len(observed))])
    R_i = np.array([((expected[i] - observed[i]) ** 2) / expected[i]
                    for i in range(len(observed))])
    R_cr = stats.chi2.ppf(q=0.95, df=len(observed)-1)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"R_i:\n{R_i}\n")

    print(f"R: {R_i.sum()}\n")

    print(f"R_cr: {R_cr}\n")

    print(f"At 0.05 satisfaction is unevenly distributed")

    ox = [i for i in range(1, len(observed)+1)]
    plt.plot(ox, observed, 'r')
    plt.plot(ox, expected, 'b')
    plt.show()


def task4():
    print("TASK 4")


    observed = np.array([109, 65, 22, 3, 1, 0])
    expected = np.array([108.6701738, 66.28880603, 20.21808584,
                         4.111010787, 0.626929145, 0.0849943876])
    R_i = np.array([((expected[i] - observed[i]) ** 2) / expected[i]
                    for i in range(len(observed))])
    R_cr = stats.chi2.ppf(q=0.95, df=len(observed)-2)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"R_i:\n{R_i}\n")

    print(f"R: {R_i.sum()}\n")

    print(f"R_cr: {R_cr}\n")

    print(f"At 0.05 accepted => Poisson distribution")

    ox = [i for i in range(1, len(observed)+1)]
    plt.plot(ox, observed, 'r')
    plt.plot(ox, expected, 'b')
    plt.show()


def task5():
    print("TASK 5")


    # mid = np.array([11,	21,	31,	41,	51,	61,	71,	81], float)
    observed = np.array([8,	7, 16, 35, 15, 8, 6, 5], float)
    expected = np.array([6.1349389, 10.69079092, 18.42332336, 22.82797103,
                         20.33937996, 13.0305589, 6.001807545, 2.551229379])

    R_i = np.array([((expected[i] - observed[i]) ** 2) / expected[i]
                    for i in range(len(observed))])
    R_cr = stats.chi2.ppf(q=0.95, df=5)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"R_i:\n{R_i}\n")

    print(f"R: {R_i.sum()}\n")

    print(f"R_cr: {R_cr}\n")

    print(f"At 0.05 random value is not distributed with normal rule")

    ox = [i for i in range(1, len(observed)+1)]
    plt.plot(ox, observed, 'r')
    plt.plot(ox, expected, 'b')
    plt.show()


def task6():
    print("TASK 6")

    observed = np.array([
        [21, 14],
        [39, 11]
    ], float)
    expected = np.array([
        [24.70588235, 10.29411765],
        [35.29411765, 14.70588235]
    ])

    R_ij = np.array([[((observed[i, j] - expected[i, j]) ** 2) / expected[i, j]
                      for i in range(len(observed))] for j in range(len(observed))]).T

    R_cr = stats.chi2.ppf(q=0.95, df=1)

    print(f"Expected:\n{expected}\n")

    print(f"Observed:\n{observed}\n")

    print(f"R_ij fail, company:\n{R_ij[0,0]}\n")
    print(f"R_ij success, company:\n{R_ij[0,1]}\n")
    print(f"R_ij fail, not company:\n{R_ij[1,0]}\n")
    print(f"R_ij success, not company:\n{R_ij[1,1]}\n")

    print(f"R: {R_ij.sum()}")

    print(f"R_cr: {R_cr}")

    print(f"At 0.05 rejected")


def task7():
    print("TASK 7")


    observed = np.array([
        [11, 8],
        [28, 24],
        [21, 19]
    ], float)
    expected = np.array([
        [10.27027027, 8.72972973],
        [28.10810811, 23.89189189],
        [21.62162162, 18.37837838]
    ])

    R_ij = np.array([[((observed[i, j] - expected[i, j]) ** 2) / expected[i, j]
                      for i in range(len(observed))] for j in range(len(observed[0]))]).T

    R_cr = stats.chi2.ppf(q=0.95, df=2)
    left = ["<= 0.5", "0.5 -1.5", "1.5+"]

    print(f"Expected:\n")
    print(f"           leq 4       more 4")

    for i in range(len(expected)):
        print(f"{left[i]: <10} {expected[i,0]} {expected[i,1]}")

    print(f"\nObserved:\n")
    print(f"           leq 4      more 4")

    for i in range(len(observed)):
        print(f"{left[i]: <10} {observed[i,0]: <10} {observed[i,1]: <10}")

    print(f"\nR_ij:\n")

    print(f"               leq 4      more 4")
    for i in range(len(R_ij)):
        print(f"{left[i]: <10} {R_ij[i,0]: 10.4f} {R_ij[i,1]: 10.4f}")

    print(f"\nR: {R_ij.sum(): .4f}")
    print(f"R_cr: {R_cr: .4f}")

    print("R < R_cr => rejected")


def task8():
    print("TASK 8")


    course = [i for i in range(1, 4)]
    activity = ["jogging", "statistics", "beer", "pop-music"]

    observed = np.array([[12,	3,	10,	18],
                         [11,	9,	10,	10],
                         [11,	9,	2,	5]], float)

    expected = np.array([[13.29090909, 8.209090909, 8.6, 12.9],
                         [12.36363636, 7.636363636, 8,	12],
                         [8.345454545, 5.154545455, 5.4, 8.1]], float)

    R_ij = np.array([[((observed[i, j] - expected[i, j]) ** 2) / expected[i, j]
                      for i in range(len(observed))] for j in range(len(observed[0]))]).T

    R_cr = stats.chi2.ppf(q=0.95, df=6)

    print(f"Observed:\n")

    print(
        f"{'': <10} {activity[0]: <10} {activity[1]: <10} {activity[2]: <10} {activity[3]: <10}")

    for i in range(len(observed)):
        print(
            f"{course[i]: <10} {observed[i, 0]: 6.4f}{observed[i, 1]: 9.4f}{observed[i, 2]: 12.4f}{observed[i, 3]: 11.4f}")

    print(f"Expected:\n")

    print(
        f"{'': <10} {activity[0]: <10} {activity[1]: <10} {activity[2]: <10} {activity[3]: <10}")

    for i in range(len(expected)):
        print(
            f"{course[i]: <10} {expected[i, 0]: 6.4f}{expected[i, 1]: 9.4f}{expected[i, 2]: 12.4f}{expected[i, 3]: 11.4f}")

    print(f"\nR: {R_ij.sum(): .4f}")
    print(f"R_cr: {R_cr: .4f}")

    print(f"R > R_cr => accept independence hypothis")


def task9():
    print("TASK 9")


    subjects = ["math", "engineering", "chemistry", "economics", "other"]
    cities = ["Moscow", "Paris"]

    moscow = np.array([95, 300, 160, 250, 320], float)
    paris = np.array([75, 200, 100, 230, 270], float)

    R_i = [1 / (moscow[i] + paris[i]) * ((moscow[i] / moscow.sum() -
                                          paris[i] / paris.sum()) ** 2) for i in range(len(paris))]

    R_cr = stats.chi2.ppf(q=0.95, df=4)

    print(f"R_i:\n{R_i}\n")

    print(f"R: {np.array(R_i).sum() * moscow.sum() * paris.sum()}\n")

    print(f"R_cr: {R_cr}")

    print(f"R > R_cr => accept hypothis that students are distributed non-equally")


def task10():
    print("TASK 10")


    pick1 = np.array([1, 5, 17, 45, 70, 51, 10, 1, 0], float)
    pick2 = np.array([1, 3, 7, 22, 88, 69, 7, 2, 1], float)

    R_i = [1 / (pick1[i] + pick2[i]) * ((pick1[i] / pick1.sum() -
                                         pick2[i] / pick2.sum()) ** 2) for i in range(len(pick1))]

    R_cr = stats.chi2.ppf(q=0.95, df=8)

    print(f"R_i:\n{R_i}\n")

    print(f"R: {np.array(R_i).sum() * pick1.sum() * pick2.sum()}\n")

    print(f"R_cr: {R_cr}")

    print(f"R > R_cr => properties are not homogeneous")


def task11():
    print("TASK 11")


    y19 = np.array([1576, 1830, 1778, 1603], float)
    y20 = np.array([1731, 1777, 2716, 4753], float)

    R_i = [1 / (y19[i] + y20[i]) * ((y19[i] / y19.sum() -
                                          y20[i] / y20.sum()) ** 2) for i in range(len(y20))]

    R_cr = stats.chi2.ppf(q=0.95, df=2)

    print(f"R_i:\n{R_i}\n")

    print(f"R: {np.array(R_i).sum() * y19.sum() * y20.sum()}\n")

    print(f"R_cr: {R_cr}")

    print(f"R > R_cr => reject homogenous hypothis")

task1()
task2()
task3()
task4()
task5()
task6()
task7()
task8()
task9()
task10()
task11()
