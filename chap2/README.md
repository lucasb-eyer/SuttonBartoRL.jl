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

2.6: Non-stationary bandits
===========================

This figure is not present in the book, but it corresponds to [Exercise 2.7](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node20.html).
It shows the same experiment as the 10-armed bandits above, except that the means of the rewards start all the same and then go on random walks.
Since the real values of actions are changing over time, both a static value estimators and estimators taking ALL samples into account will not follow this movement and get worse over time or stagnate, respectively.
What works well in this case, is a decaying estimate, giving old samples less importance the older they get.
This mechanism is almost identical to the concept of "momentum" in optimization algorithms.

<p align=middle><img src=plots/2.6.png/></p>

2.7: Optimistic value estimators
================================

This is [Figure 2.4 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node21.html).
It describes a pretty neat trick to greatly improve even the greedy algorithm:
Initialize the value estimators in a way that they higly overestimate the value of *every* action.
The algorithm will thus almost always be "disappointed" by the reward and move to another action, until the value estimate are better.
This encourages exploration as long as the estimates are off, and exploitation once they are good.

<p align=middle><img src=plots/2.7.png/></p>

2.8: Reinforcement comparison
=============================

This is [Figure 2.5 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node22.html).
Instead of estimating a value for actions, this keeps track of its "preference" for actions, increasing it on good reward and decreasing it on bad reward.
Whether a reward is good or bad is determined using a geometric average (momentum) of recent past rewards.
The "soft" is an alternative updating preferences slower the more preferred they get.

<p align=middle><img src=plots/2.8.png/></p>

2.9: Pursuit methods
====================

This is [Figure 2.6 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node23.html).
I couldn't say it more concisely than the book itself:

> Pursuit methods maintain both action-value estimates *and* action preferences,
> with the preferences continually "pursuing" the action that is greedy according to the current action-value estimates.

The "preference pursuit" is my attempt at solving Exercise 2.13/14:
creating a pursuit method that doesn't update probabilities but rather preferences, just as stated in the above quote.

<p align=middle><img src=plots/2.9.png/></p>
