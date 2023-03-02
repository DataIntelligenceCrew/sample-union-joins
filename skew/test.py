'''
''Estimated join sizes:
[299861334729684, 222138932504832, 249194386208400]
Estimated overlap sizes: 
[[0, 1], [0, 2], [1, 2], [0, 1, 2]]
[197966705130, 0, 182478556080, 0]
[[299663368024554, 197966705130, 0], [221758487243622, 380445261210, 0], [249011907652320, 182478556080, 0]]
Estimated union size: 
770814208181706.0
Exact join sizes:
[19049254, 103694, 716760]
Exact overlap sizes:
[1133, 1466, 1289, 577]
Exact union size:
11105
'''

import matplotlib.pyplot as plt
import numpy as np
import math

estimated_sizes = [299861334729684, 222138932504832, 249194386208400]
estimated_overlaps = [197966705130, 0, 182478556080, 0]


exact_sizes = [19049254, 103694, 716760]
exact_overlaps = [1133, 1466, 1289, 577]

def calc_As(Os, ans, e_j):
    n = len(e_j)
    As = [ [0]*n for i in range(n)]
    for j in range(len(e_j)):
        As[j][n-1] = Os[len(Os)-1]
        for k in range(n-1, 0, -1):
            A = 0 
            count = 0
            for index in range(len(ans)):
                if (len(ans[index]) == k) and (j in ans[index]):
                    A += Os[index]
                    count += 1
            if (k == 1): A += e_j[j]
            for r in range(k+1, n+1):
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    print("As: ", As)
    return As


def calc_U(Js, Os, ans):
    As = calc_As(Os, ans, Js)
    U = 0
    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
    print("U: ", U)
    return U

est_union = 770814208181706.0
exact_union = 19866397.0

# randomly generate three numbers that are smaller than 1 and greater than 0 according to the estimated sizes
random = np.random.rand(3)

ratio_guess = [x / sum(estimated_sizes) for x in estimated_sizes]
# print ratio of estimated and exact sizes over the unions
print([x / est_union for x in estimated_sizes])
print([x / exact_union for x in exact_sizes])

# print(calc_U(exact_sizes, exact_overlaps, [[0, 1], [0, 2], [1, 2], [0, 1, 2]]))

# draw figure that compares the ratio of estimated and exact sizes of the union
def compare_ratios():
    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize =(12, 8))
    # set height of bar
    bars1 = [estimated_sizes[0], estimated_sizes[1], estimated_sizes[2]]
    bars2 = [exact_sizes[0], exact_sizes[1], exact_sizes[2]]
    bars3 = [ratio_guess[0], ratio_guess[1], ratio_guess[2]]
    bars4 = [random[0], random[1], random[2]]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    # Make the plot
    plt.bar(r1, [x / est_union for x in bars1], color ='#7f6d5f', width = barWidth,
            edgecolor ='grey', label ='Estimated')
    plt.bar(r2, [x / exact_union for x in bars2], color ='#557f2d', width = barWidth,
            edgecolor ='grey', label ='Exact')
    plt.bar(r3, bars3, color ='#0d261c', width = barWidth, edgecolor ='grey', label ='Ratio Guess')
    plt.bar(r4, bars4, color ='#2d7f5e', width = barWidth, edgecolor ='grey', label ='Random Guess')
    # Add xticks on the middle of the group bars
    plt.xlabel('Tables', fontweight ='bold')
    plt.ylabel('Ratio', fontweight ='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))],
            ['Table 1', 'Table 2', 'Table 3'])
    # Create legend & Show graphic
    plt.legend()
    plt.show()

compare_ratios()