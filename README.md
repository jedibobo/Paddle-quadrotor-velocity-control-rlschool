# quadrotor-velocity-control-rlschool

## reinforcement learning using PARL and Quadrotor env from rlschool

## tools used

Python: Python 3.6+

[paddlepaddle](https://github.com/PaddlePaddle/Paddle)

[PARL](https://github.com/PaddlePaddle/PARL)

[rlschools](https://github.com/PaddlePaddle/RLSchool)

![image](https://github.com/jedibobo/quadrotor-velocity-control-rlschool/blob/master/imgs/demo_velocity_control.gif)

## To Install prerequisites

> > pip install -r requirements.txt

## Train

> > python train.py

The program will evaluate itself and print the evaluation results in the terminal.

## Restore previous model

**use function restore_model in train.py**
for example:add line in main()

> > restore_model(570684)

note that if the models in QuadrotorModel is modified, you cannot use previously trained models.

## results

**_Because time is limitted, run process are not totally finished._**

![image](https://github.com/jedibobo/quadrotor-velocity-control-rlschool/blob/master/results/2020-06-30%20090011.png)
