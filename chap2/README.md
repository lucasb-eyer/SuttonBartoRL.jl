2.2: 10 Armed bandits
=====================

This is [Figure 2.1 in the book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node16.html).
It shows the average (over 2000 runs) reward and optimal actions picked.
The bandits are 10-armed, each arm giving a normally distributed random reward whose mean has been (randomly) fixed at the start.
The action values are estimated by the sample means.
The main point of this plot is that you need exploration to perform well on such a stochastic task.

<p align=middle><img src=plots/2.2.png/></p>
