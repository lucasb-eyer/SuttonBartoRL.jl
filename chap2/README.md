2.2: 10 Armed bandits
=====================

This is [Figure 2.1 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node16.html).
It shows the average (over 2000 runs) reward and optimal actions picked.
The bandits are 10-armed, each arm giving a normally distributed random reward whose mean has been (randomly) fixed at the start.
The action values are estimated by the sample means.
The main point of this plot is that you need exploration to perform well on such a stochastic task.

<p align=middle><img src=plots/2.2.png/></p>

2.4: Hard binary bandits
========================

This is [Figure 2.3 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node18.html).
The top part (A) corresponds to a binary bandit where both possible actions have low probability (0.1 and 0.2) to give reward,
while for the bottom part (B) both actions have high probability (0.8 and 0.9) to give reward.
This means that when one observes the reward of an action, one can't infer anything about the other action.
But this is what supervised algorithms do (in principle), and we can see they don't perform well.

<p align=middle><img src=plots/2.4.png/></p>
