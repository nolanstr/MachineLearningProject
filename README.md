# MachineLearningProject
Code for running my CS 6350


I recommend  running code within a Linux or Mac terminal or using GitBash with
windows. If the code does not run, make sure to update your $PYTHONPATH
enviorment variable so that when you run the code with Python it looks in the
right location for the bingo package. 

To run code for each seperate version, change into the respective directory.

Standard GPSR -> bingo_runs
Probabilistic GPSR -> smc_bingo_runs
Oscillating GPSR -> oscillating_runs

From these directories, run the following line.

```python
python run_bingo.py
```

It is important to note that these were ran on CHPC with 8 CPU cores and thus
computation times of those reported will most likely differ with any times you
find. In order to observe all the models in the pickles, run the following line
of code.

```python
python pull_data.py
```

Thanks!
