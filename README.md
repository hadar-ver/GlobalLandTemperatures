# GlobalLandTemperatures
Predict global land temperatures by country using LSTM.

This work focuses on application of modern Machine Learning approach of Deep Recurrent Neural Network techniques, to model and predict general time series problem obtained from Climate Change: Earth Surface Temperature Dataset collected from the Kaggle machine learning repository .  The selection of the data that were used in the experiments was focused on land temperature, intending to predict their evolution over time. In this work, 3 models will be used to predict the temperature: Persistence model (as base line), deep (and simple) RNN and Deep LSTM. The model will be trained, tuned, and evaluated on the same train data dets and then will be checked once on the same test set. All the preparation, validation and test-set creation are unique for time series problems and will be explained in detail in this work.


## •	DATA COLLECTION:
The data understanding phase starts with an initial data collection and proceeds with activities to get familiar with the data, to identify data quality problems, to discover first insights into the data, or to detect interesting subsets to form hypotheses for hidden information. There is a close link between Business Understanding and Data Understanding. The formulation of the data mining problem and the project plan require at least some understanding of the available data [2].
For this project, Climate Change: Earth Surface Temperature dataset, from the Kaggle Repository were used. The data contains 577K records that were collected for 270 years between 1743 and 2013 in 238 countries around the world and features of: (1)Time stamp (equal interval of 1 month between each sample), (2)average temperature- this feature is the target feature (3) uncertainty level of this temperature (4) country.
 Early data was collected by technicians using mercury thermometers, where any variation in the visit time impacted measurements. In the 1940s, the construction of airports caused many weather stations to be moved. In the 1980s, there was a move to electronic thermometers that are said to have a cooling bias. Given this complexity, there are a range of organizations that collate climate trends data. The three most cited land and ocean temperature data sets are NOAA’s MLOST, NASA’s GISTEMP and the UK’s HadCrut. In this dataset Kaggle has repackaged the data from a newer compilation put together by the Berkeley Earth, which is affiliated with Lawrence Berkeley National Laboratory. The Berkeley Earth Surface Temperature Study combines 1.6 billion temperature reports from 16 pre-existing archives. It is nicely packaged and allows for slicing into interesting subsets (for example by country).
To predict global temperature the data set grouped by time stamp, which yield to new data set with 1860 records (monthly timestamps). The data (as shown in the figures below) shows that the global average temperature went up in a very sharp way between 18th and middle 19th century.

![image](https://user-images.githubusercontent.com/71387302/198237463-e8a01dbf-30d4-461d-acde-c7c612dbfa88.png)

It appears that before 1857 data mostly represent countries with low average temperature and not represent well the global average temperature. To support the task of global temperature prediction I have used data from 1857; In the graphs below we can observed increase in global average temperature from 17.58 ⸰C to 19.50 ⸰C in the last 2 centuries (1857-2012). It also shows that the uncertainty decreases over time.

![image](https://user-images.githubusercontent.com/71387302/198237562-2a4a69b6-fa64-461b-ada3-baa6ec6f3f07.png)


## •	DATA PREPARATION:
In the “Data Preparation” phase I have collected the relevant data and prepared it for the actual data mining task.  This includes the pre-processing, e.g.  data reduction and filtering, as well as feature generation with respect to the data mining project goal. The data preparation phase covers all activities to construct the final dataset (data that will be fed into the modelling tool(s)) from the initial raw data [2].
The dataset I have used for this project is a time-series tabular dataset that needs to be well prepared to fit to the prediction task and to the relevant models.
These are the preparations I have performed on the aggregate dataset:
### 1.	Null values imputation:
In almost every domain from industry to biology, finance up to social science different time series data are measured. While the recorded datasets itself may be different, one common problem are missing values. Missing values in time-series problem need to be carefully imputed by using the past data only to avoid data leakage from future to past. The method I have used for missing data imputation is rolling average of the past 20 years from a missing datapoint. The missing values were checked and imputed per country and not worldwide.
### 2.	Stationarity check:
Additional important data understanding stage in time-series problems is stationary check. Applying a model on Stationary data leading to better performances so it’s very important stage. In the most intuitive sense, stationarity means that the statistical properties of a process generating a time series do not change over time. It does not mean that the series does not change over time, just that the way it changes does not itself change over time.  The algebraic equivalent is thus a linear function, perhaps, and not a constant one; the value of a linear function changes as 𝒙 grows, but the way it changes remains constant. Stationarity was checked with ADF test: The intuition behind the test is that if the series is characterized by a unit root process (unit root is like other root cause that has greater effect on the records over time) then previous time stamp will provide no relevant information in predicting the change in current time stamp. In this case the null hypothesis is not rejected. In contrast, when the process has no unit root, it is stationary and hence exhibits reversion to the mean - so the lagged level will provide relevant information in predicting the change of the series and the null hypothesis of a unit root will be rejected. The results shows that the ADF value is lower than the critical values of 1%, 5% and 10%- which means that we didn’t reject the null hypothesis and the data is non- stationary.

![image](https://user-images.githubusercontent.com/71387302/198237743-410c802b-3402-4bae-b7de-27ce6da9e275.png)

### 3.	Train-test split:
In time series problem the split to train, validation and test set is not random and it based on the time that the event occurred. The latest 10% of the records were allocated for test set. The train-validation sets will be generated in the cross-validation method describes in the ‘forecasting method’ section
### 4.	New feature:
Lagging Transformation- as a result of ADF test, the data is not-stationarity and it can lead to poor performances. One of the methods to overcome the non-stationarity problem is to give representation to previous 12 months as features so the model will learn from these features as well.

## •	MODELS:
#### BASE LINE MODEL: 
Establishing a baseline is essential on any time series forecasting problem. A baseline in performance gives an idea of how well all other models will perform on the problem. For this purpose, I have used persistence forecast, which is Naïve, simple, fast, and repeatable forecast technique and one of the common baseline models for time-series. The persistence forecast uses the value at the previous time step (t-1) to predict the expected outcome at the next time step(t).
### RNN & LSTM:
Recurrent Neural Network has brought an interesting aspect in basic neural networks. The simplest type of RNN is the Vanilla Recurrent Neural Network which takes a predetermined size of input vector of features. The Artificial neural networks (ANNs) are not able to capture the sequential information from the input data so in sequential problems like Time-series Forecasting where it's required to predict the next future value by using previous historical sequential data, the Artificial Neural Networks get failed. Thus, RNN came into existence because RNNs are a kind of neural network where the output from previous step is fed as input to the current step. The most important feature of RNN is the feedback network in hidden state, which remembers some information from the previous sequence.
The output of a Recurrent Network at time step t is affected by the output of time step t-l. The Recurrent network takes two inputs, the current time step input Xt and the hidden past state output St-l.  
Here, St is Next hidden state output; W, U, V are Weights and Ot is Output of current state. tanh is the default activation function that is used in the Recurrent network to squeeze the values between −1 to 1. RNN works well with moderated size data but as the size of data samples increases, RNN as the network becomes dense network fails to propagate the information from the previous state, this is known as short memory problem of the Recurrent neural networks. In RNNs information travels through time which means the information from previous time step is used as the input for the next time step by which cost function or error is calculated at each time step. When the network becomes large and dense the gradient starts to become small due to which weights don't get updated in the network, this problem is known as the Vanishing Gradient Problem.
Due to these drawbacks of Recurrent Neural Network, LSTM comes into existence which solves the short memory and vanishing gradient problem with its special gated network cells.
Long Short-Term Memory networks were firstly introduced by Hoch Reiter & Schmid Huber in 1997. LSTMs are a specific kind of Recurrent Neural Networks (RNNs), designed to solve the vanishing gradient problem, by preserving the error that can be backpropagated through time and layers. Moreover, LSTMs overcome the weakness of learning long-range dependencies, a trait that is essential in time-series forecasting. Unlike traditional neural networks, an LSTM’s layer consists of a set of blocks, recurrently connected. Each block contains one or multiple recurrently connected memory cells and three multiplicative units – the input, output and, forget gates. The memory cell remembers values over arbitrary time intervals, while the three gates arrange the information flow into and out of the cell. The cell stores the desired information, by performing acts of reading, writing, and erasing, via the gates that open and close. These extra interactions among the four elements of memory blocks, make LSTMs capable of learning which data are important to keep, and which are not. Gates are associated with sets of weights, which filter the information blocking it, or allowing it to pass. Weights are not fixed but adjust over the recurrent learning process. Thus, relevant information can pass down the chain of sequences to make predictions, achieving to maintain and represent long-term data.
##### DEEP RNN'S:
The architecture of deep-RNN is stacking multiple RNN layers together into a ‘deep’ architecture. 
In deep-RNNs, the sharing states are decomposed into multiple layers to gain nice properties from ‘deep’ architectures. Experimental evidence suggest the significant benefit of building RNNs with ‘deep’ architectures. The unfolding topological graph is presented in the figure below to demonstrate the working process of a deep-RNN with N layers. RNN aims to map the input sequence of x values into corresponding sequential outputs: y. The learning process conducted every single time step from t=1 to t=T . For time step t , the network neuron parameters at lth layer update its sharing states. Due to the sharing properties of RNNs, the algorithm is thus capable to learn uncertainties repeated in previous time steps. 
Deep LSTM- network is a Deep RNN network boosted with LSTM units instead of a Simple RNN unit 

![image](https://user-images.githubusercontent.com/71387302/198238734-2f01055d-54fa-4f68-acb7-ee017d8e1daf.png)


## •	CROSS VALIDATION:
Cross validation in the case of time series is not trivial; We cannot choose random samples and assign them to either the test set or the train set because it makes no sense to use the values from the future to forecast values in the past. In simple word we want to avoid future-looking when we train our model. There is a temporal dependency between observations, and we must preserve that relation during testing. 
The method that can be used for cross-validating the time-series model is cross-validation on a rolling basis. Start with a small subset of data for training purpose, forecast for the later data points and then checking the accuracy for the forecasted data points. The basic method for time series CV is Time Series Split Cross-Validation which means that the same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted. However, this may introduce leakage from future data to the model. The model will observe future patterns to forecast and try to memorize them. To overcome this issue, in this project I used ‘Block Cross-Validation’ method.  It works by adding margins at two positions. The first is between the training and validation folds to prevent the model from observing lag values which are used twice, once as a regressor and another as a response. The second is between the folds used at each iteration to prevent the model from memorizing patterns from an iteration to the next.

![image](https://user-images.githubusercontent.com/71387302/198238884-80befad2-1e54-42bf-8261-46540aa8b29a.png)

## 6.	RESULTS EVALUATION
Within the subsequent “Result Evaluation” phase the trained model is tested against the validation folds that generated in the “modelling” stage, and the results are assessed  according  to  the  chosen metric which underlying  business  objectives.  One of the key questions in the forecasting process has to do with the measuring of the forecast accuracy. There is a very long list of metrics that different businesses use to measure this forecast accuracy. Mean Absolute Percent Error (MAPE) is a very commonly used metric for forecast accuracy.  The MAPE formula is:   ![image](https://user-images.githubusercontent.com/71387302/198239240-85f10a36-34f9-4dd4-8fbb-59efa075b546.png)

Since MAPE is a measure of error, high numbers are bad and low numbers are good. For reporting purposes, some companies will translate this to a score by subtracting the MAPE from 100 . 
the formula is: 100-MAPE, this transformation is easy to explain and more intuitive for most of the people,  and it does not depend on scale. For that reasons, this is the chosen metric to assess the model’s performances. 

Initial results:
![image](https://user-images.githubusercontent.com/71387302/198240140-000491c9-3795-4b7b-961c-85605a5c377d.png)

After successful evaluation of the trained model, it is deployed into production in the “Deployment” phase. However, the deployment also requires a stable set-up for data acquisition including a hyper parameter tuning and data processing infrastructure. 

## •	OPTUNA- HYPER PARAMETER TUNING
Hyperparameter search is one of the most cumbersome tasks in machine learning projects. The complexity of deep learning method is growing with its popularity, and the framework of efficient automatic hyperparameter tuning is in higher demand than ever. Optuna was introduced by Takuya Akiba et. al. [8] in 2019. Optuna is an open-source python library for hyperparameter optimization. In the background, Optuna aims to balance the sampling and pruning algorithms for relational parameter sampling which aims to exploit the correlation between parameters. Likewise, Optuna implements a variant of the Asynchronous Successive Halving (ASHA) [9] algorithm for the pruning of search spaces. Optuna emerges as a hyperparameter optimization software under a new design-criteria which is based on three fundamental ideas: define-by-run API which allows users to construct and manipulate search spaces in a dynamic way, efficient implementation that focuses on the optimal functionality of sampling strategies as well as pruning algorithms, and easy-to-setup that focuses on versatility, that is, it allows optimizing functions in light environments as well as large-scale experiments in environments based on distributed and parallel computing. 
In this project I ran 50 trials with different hyper parameters combinations, trained the networks again (both LSTM and RNN), predict validation set values and evaluate the results. In the figures below the left plot presented all scores in each one of the ‘Optuna’ trials, while the right plot presents the average scores of the ensembled validated sets after the hyper parameter tuning. Boths scores of RNN and LSTM were improved as a result if the hyper parameter tuning. Eventually the best hyper parameter was chosen and implemented on a test set.

![image](https://user-images.githubusercontent.com/71387302/198240466-1ca9c349-8a2a-484e-8cbf-ab93310f4a81.png)

![image](https://user-images.githubusercontent.com/71387302/198240595-b1367ff4-b2da-49ab-ba94-c45997cced0b.png)

## •	FINAL RESULTS- TEST SET PREDICTION AND EVALUATION
The ‘Test’ dataset provides the gold standard used to evaluate the model. It is only used once a model is completely trained (using the train and validation folds). The test set is generally what is used to evaluate competing. In time series problem the test set obtained from the last X periods of time. In these case- 186 months.
Auto-regressive process to build new ‘Test’ set: The features of the data contain information of the previous 12-time stamps, in the case of the test set we should not have this data available (except for the first sample which contains data from the last 12 real values of the train set) the ‘new’ test set were built in an auto regressive process where the features were generated from latest predictions and not the real test data set. Based on these generated features, the predictions are being made. This method prevents data leaks from training to test set and it provides predictions that are much closer to the results we will get in production when our data will not be seen.
![image](https://user-images.githubusercontent.com/71387302/198240913-fb06651e-679c-431f-802a-b9fc389f9882.png)



The results are great! Score of 95.7% for Deep LSTM model, 15.6% above the baseline.
















