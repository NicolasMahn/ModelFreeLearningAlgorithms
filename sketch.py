import matplotlib.pyplot as plt


def showList(data, title="Fitness Curve", subtitle="", x_label="Episodes", y_label="Return"):
    # fig = plt.figure()
    plt.suptitle(title, fontsize=18)  # title
    plt.title(subtitle, fontsize=10)  # subtitle
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot([data[i] for i in range(0, len(data))], color="black")
    plt.show()
