# COVID-NSF: Custom PPO algorithm with multi-threading for data collection implemented in TensorFlow

The code repository contains the implementation of PPO clip algorithm, which is a popular reinforcement learning algorithm, written in TensorFlow. Reinforcement learning algorithms typically have three stages: data collection, finding gradients, and updating the policy network. The last two stages are typically done using deep learning packages such as TensorFlow and PyTorch and are implemented on a GPU. However, the first stage, data collection, is typically done on a CPU, which can slow down the overall process.

To overcome this limitation and improve the overall speed of the algorithm, I have used multi-threading through the Concurrent Python toolbox. Multi-threading allows the data collection process to be executed simultaneously with the other stages of the algorithm, making the overall process more efficient. 

Additionally, the code repository also includes the necessary functions and classes to implement the PPO algorithm, such as the actor-critic network, the loss function, and the optimizer. The code is organized in a clear and easy-to-understand manner, making it easy for others to understand and adapt for their own use.

Overall, the use of multi-threading and the TensorFlow library makes this implementation of the PPO algorithm highly efficient and suitable for large-scale reinforcement learning tasks.

This is  used for computing the optimal policies in SEIR epidamic models. During the pandemic different control policies such as Lockdown, social distancing, and Congregation restriction has been implemented in an ad-hoc manner to control the disease spread while minimizing the economic/social losses. However, the control policies should be designed to balance their effectiveness on disease spreading while minimizing their impacts on economy/society, poses the key challenge in pandemic-time decision making.  I study and propose the optimal control policies through Reinforcement Learning under data uncertainity and sampling bias. To achieve this I have used the Proximal Policy Optimization algorithm, implemented using TensorFlow deeplearning toolbox. 


`In summary, I have created a custom implementation of the PPO algorithm in TensorFlow that utilizes multi-threading for data collection. The use of multi-threading improves the efficiency of the algorithm and the TensorFlow library allows for easy implementation of the PPO algorithm's necessary components such as the actor-critic network, loss function, and optimizer. This implementation is intended for use in large-scale reinforcement learning tasks, specifically in the context of pandemic decision making and finding optimal control policies for disease spread.`
