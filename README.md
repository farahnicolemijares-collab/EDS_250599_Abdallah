# EDS_250599_Abdallah
**BIO-01: Heart Rate Recovery Analysis**
Course: Computer Programming 1 | Academic Year: 2026
Student: Farah Nicole Abdallah | ID: 25-0599

## Project Description
This project implements an automated Object-Oriented 
Python data pipeline for analyzing Heart Rate Recovery 
(HRR) dynamics from real-world wearable running sensor 
data. The pipeline ingests, cleans, and statistically 
analyzes 3,577 true recovery events using NumPy-based 
descriptive statistics, IQR outlier detection, Pearson 
correlation, and comparative group analysis. The system 
deploys five automated visualizations including static 
charts and animated outputs to characterize cardiac 
recovery behavior across Low HRR and High HRR groups.

## Dataset
Running and Heart Rate Data
https://www.kaggle.com/datasets/mcandocia/running-heart-rate-recovery
File used: s1_stop_to_start.csv
Unique Filter: rest_time > 0 (true recovery events only)

## How to Run
pip install -r requirements.txt
python main.py

## Outputs
All plots and animations are saved to the outputs/ folder.
- plot1_histogram.png
- plot2_boxplot.png
- plot3_heatmap.png
- animation1_rolling_mean.gif
- animation2_scatter_cluster.html
