# probabilistic forecast for a sub-9:00 1.5 mile run

## the motivation

*started july 28, 2025:*

i wanted to answer one question:
can i run 1.5 miles in under 9:00 by august 15, 2025?

this repo is the day-to-day, real-time, transparent data science log of that goal

i will be giving daily updates on my progress and allow you to see it as well in the data/weights folder
you can view the run time dataset in the /data/runtimes.json file

i thought it would be fun to make a model that can predict whether i will or not and find out the results on the final day

there is an update log on the bottom to read daily updates

---

## scientific method:

- observation: i was realizing as i got closer to august 15, 2025 my chances of reaching a sub 9:00 1.5 mile run was achievable using the strava app and seeing the progress

- question: i wanted to know if i could make a prediction model that would take in my running times and basically predict my odds of achieving this goal

- research: i simply plotted my times on a chart and realized the data was heteroscedastic meaning the variance was not going to equal throughout. i did some digging and learned about weighted least squares which lead to the idea of using a time-weighted regression model. and i had to simulate the circumstances so decided on using monte carlo simulations.

- hypothesis: if i was able to use these methods correctly, i could make a working forecasting model to see the odds i get a sub 9:00 1.5 mile. obivously i was able to do that but the real question was would the predictions of the model hold up as i progressed to august 15, 2025 and would they intuitively feel true?

- experiment: after a couple hours of converting the equations into code (as seem in runprediction.py) and creating visuals i was able to create such a system. now i just need to acquire more data to make the model more accurate as it gets closer to the end date which has encouraged me to run daily (hopefully).

- test hypothesis: too early to tell

- conclusion: experiment not concluded

---

## what is this?

- a time-weighted regression and monte carlo simulation forecasting my odds of a sub-9:00 1.5-mile
- updated daily with new runs, results/weights, and probabilities
- demonstrates real analytics, reproducible code, and version-controlled personal improvement
- encourages me to run everyday to collect more data to make the model more accurate to encourage to run more to collect more data which hopefully allows me to get a sub 9:00 1.5 mile run

---

## the data

- all runs are in `data/runtimes.json` (date, distance, time) 
- weights for every run are logged daily in `data/weights/YYYY-MM-DD_weights.json`
- predictions are logged in the data/predictions

example data:
{"date": "2025-07-29", "distance": 1.24, "time": "7:29"}

---

## method

- **exponentially weighted regression:** recent runs have more influence
- **gaussian distance weighting:** runs closer to 1.5 miles matter more. shorter or much longer runs have less weight
- **monte carlo simulation:** forecasts thousands of future outcomes to estimate probability and uncertainty for hitting sub-9:00

---

## key equations

- recency weight: exponential decay based on how many days ago the run was
- distance weight: bell curve (gaussian) centered at 1.5 miles
- combined weight: product of recency and distance weights
- weighted regression: finds the best-fit line using these weights
- forecast: simulates future possible results based on model and uncertainty

---

## code structure

- main script: `runprediction.py`
- data: `data/runtimes.json`
- weights: `data/weights/`
- predictions: `data/predictions`
- requirements: `requirements.txt`

---

## day-to-day philosophy

update data, model, and forecasts daily until August 15, 2025
this repo is a transparent, honest log of training and self improvement with the numbers to back it up
ironically i need more data which means i have to run more, which in itself making this model has increased me running more to get more data

---

## example calculation

on july 29, 2025, i ran 1.24 miles in 7:29.  
recency weight for that day: 1.0  
distance weight for 1.24 miles: about 0.81  
combined weight: 0.81  
this run influences the model accordingly since I didn't run the full 1.5 miles and gave a weight scoring it less. since running shorter distances allows you to exert more energy into moving faster

---

## update log

*july 29, 2025, 23:00*

when i was running today, i started my first two laps with pace but reveille started playing. in the military you have to stop what you are doing and stand at attention, so being outside running i had to stop and stand at attention. i gave myself a 5 minute break and ran for 1.24 miles clocking in at 7:29. 

i realized two things today: 

- i needed to weigh the distance i ran, so i added the weighted guassian bell curve with max score at 1.5 miles ran
- i needed to present the data chronologically so i made the data/weights and data/predictions folder to help visualize the data as the days progressed (prob make the data/predictions folder and files tomorrow)

---

## how to run

1. clone the repo  
   `git clone https://github.com/westagain/run-prediction-model.git`

2. install requirements  
   `pip install -r requirements.txt`

3. run the script  
   `python runprediction.py`

4. update your data  
   just edit `data/runtimes.json` with your new run (add a new line in the same format as the others)

5. check out the results  
   - predictions, visuals, and weights get updated automatically
   - open the latest files in `data/weights/` to see how each run is weighted
   - see your updated probability in the terminal output and in the plot

that’s it.  
just run, update, and repeat.

---

## license

MIT

---