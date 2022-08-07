# BARC System Identification

This repo has the code to perform system identification of the lateral dynamics of BARC car. 
To collect data, two nodes need to be running, vive_server_client_node.py and id_data_node.py. 
The Vive node should always be running. For each step, open id_data_node and edit the steering command, then re-run it.
Once all data is collected, use process_id_data.py script to get the system parameters (Popt).

For validation, change the Popt in the functions "beta_to_pwm," "beta_to_angle," and "angle_to_pwm" within verification_node.py. 
The process for acquiring data will be similar to before, except, instead of a PWM command, give the desired steering angle. 
So, for each step, open verification_node and edit the steering angle command, then re-run it.
Once all data is collected, use process_validation_data.py script to get the error in the model.
