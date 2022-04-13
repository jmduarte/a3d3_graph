import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from collections import OrderedDict


def nonlinear(x):
    # return 6*np.log10(x)
    return 0.05 * np.power(np.log10(x), 4)
    # return 1e-4*np.power(x,1/2)


# Model name -> Dataset, Accuracy, MACs, BOPs, Weights, Total weight bits
input_dict = OrderedDict(
    [
        (
            "MobiletNet-w4a4",
            ["ImageNet", 71.14, 557381408, 74070028288, 4208224, 16839808],
        ),
        ("CNV-w1a1", ["CIFAR-10", 84.22, 57906176, 107672576, 1542848, 1542848]),
        ("CNV-w1a2", ["CIFAR-10", 87.80, 57906176, 165578752, 1542848, 1542848]),
        ("CNV-w2a2", ["CIFAR-10", 89.03, 57906176, 331157504, 1542848, 3085696]),
        ("TFC-w1a1", ["MNIST", 93.17, 59008, 59008, 59008, 59008]),
        ("TFC-w1a2", ["MNIST", 94.79, 59008, 118016, 59008, 59008]),
        ("TFC-w2a2", ["MNIST", 96.60, 59008, 236032, 59008, 118016]),
    ]
)

labels = input_dict.keys()
y = np.array([input_dict[key][1] for key in input_dict])
x = np.array([input_dict[key][3] for key in input_dict])
w = np.array([input_dict[key][5] for key in input_dict])

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#7f7f7f",
    "#bcbd22",
    "#d62728",
]

plt.style.use([hep.style.ROOT, hep.style.firamath])
# hep.set_style("CMS")
# hep.set_style({"font.sans-serif":'Comic Sans MS'})


f, axs = plt.subplots(3, 1)
for xi, yi, wi, l, c in zip(x, y, w, labels, colors):
    if "TFC" in l:
        ax = axs[2]
    elif "CNV" in l:
        ax = axs[1]
    else:
        ax = axs[0]
    ax.plot([xi], [yi], label=l, marker="o", markersize=nonlinear(wi), color=c)
    ax.plot([xi], [yi], label=l, marker="*", markersize=10, color="white")
    if "TFC" in l:
        ax.text(xi * 1.7, yi - 0.3, l, color=c)
    elif "CNV" in l:
        if "w2a2" in l:
            ax.text(xi * 5, yi - 0.5, l, color=c)
        else:
            ax.text(xi * 0.005, yi - 0.5, l, color=c)
    else:
        ax.text(xi * 0.0002, yi - 3, l, color=c)

axs[0].plot(
    [1e5],
    [75],
    label="100 k",
    marker="o",
    markersize=nonlinear(100e3),
    color="gray",
)
axs[0].plot(
    [5e5],
    [75],
    label="1 M",
    marker="o",
    markersize=nonlinear(1e6),
    color="gray",
)
axs[0].plot(
    [1e7],
    [75],
    label="10 M",
    marker="o",
    markersize=nonlinear(10e6),
    color="gray",
)

axs[0].text(5e4, 77.8, "Weight bits", color="gray", size=20)
axs[0].text(1e5, 75, "100 kb", color="white", size=10, ha="center", va="center")
axs[0].text(5e5, 75, "1 Mb", color="white", size=18, ha="center", va="center")
axs[0].text(1e7, 75, "10 Mb", color="white", size=18, ha="center", va="center")


xmin = 2e4
xmax = 2e12
for ax in axs:
    ax.semilogx()

axs[0].set_xlim(xmin, xmax)
axs[0].set_ylim(65, 80)
axs[1].set_xlim(xmin, xmax)
axs[1].set_ylim(81, 92)
axs[2].set_xlim(xmin, xmax)
axs[2].set_ylim(92, 98)
axs[2].set_xlabel("BOPs")
axs[1].set_ylabel("Accuracy [%]\nCIFAR-10")
axs[0].set_ylabel("\nImageNet")
axs[2].set_ylabel("\nMNIST")

axs[0].axes.xaxis.set_ticklabels([])
axs[1].axes.xaxis.set_ticklabels([])

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig("qonnx_graph.pdf")
plt.savefig("qonnx_graph.png")
