# Demo Model Predictive Control (MPC) Scripts

These scripts demonstrate how to implement MPC in Python using CasADi. There
are two versions:

* `normal.py` - point tracking for a robot with unicycle dynamics
* `obstacle_avoidance.py` - same as `normal.py` but with static obstacle
avoidance

I created these as part of a guest lecture I gave for EGRE 691 - Autonomous
Physical Systems.

# Usage

The scripts require the following packages

* `numpy`
* `matplotlib`
* `casadi`

To run, invoke the following:

```shell
python3 normal.py
```
or
```shell
python3 obstacle_avoidance.py
```
