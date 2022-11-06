import matplotlib.pyplot as plt


def showList(data, title="Fitness Curve", subtitle="", xLabel="Episodes", yLabel="Return"):
    # fig = plt.figure()
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.plot([data[i] for i in range(0, len(data))], color="black")
    plt.show()
