import os 
os.system("pip install torch")
os.system("pip install --force gradio==3.28.3")
import torch
import huggingface_hub
huggingface_hub.hf_hub_download(repo_id="victor29/EarlyHeartBeat_v1",filename="MedicalNet.pt",repo_type="space")
import random
import torch.nn.functional as F
import math
import ctypes
import gradio as gr
import pickle

#You are not that good!
class MedicalNet(torch.nn.Module):

    def __init__(self):
        super(MedicalNet,self).__init__()
        self.MedicalNet = torch.nn.ModuleList()
        self.MedicalNet.append(torch.nn.Linear(529,100))
        self.MedicalNet.append(torch.nn.Linear(100,100))
        self.MedicalNet.append(torch.nn.Linear(100,100))
        self.MedicalNet.append(torch.nn.Linear(100,1))
        self.LossFn = torch.nn.MSELoss()
        self.losses = []
        self.MAES = []
        self.Accuracies = []
        self.epoch = 0
        self.positives = 0
        self.accuracy = 0
        self.diff = 0
        self.quot = 0
        self.sum = 0
        self.MAE = 0
        #Wow, that sucks
    def forward(self,datapoint):
        x = datapoint
        layers = len(self.MedicalNet)
        for layer in self.MedicalNet[:len(self.MedicalNet)-1]:
            x = F.relu(layer(x))
        x = torch.sigmoid(self.MedicalNet[layers-1](x))
        return x

    def model(self):
        return self.MedicalNet

    def learn(self,X,Y,testingmode=False):
        optimizer = torch.optim.Adam(self.MedicalNet.parameters(),lr=0.01)
        loss = self.LossFn(X,Y)
        if testingmode==False:
            self.losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.epoch += 1
        action = round(X.item(),0)
        if action == Y.item():
            self.positives += 1
        self.accuracy = float(self.positives/self.epoch)
        self.Accuracies.append(self.accuracy)
        self.diff = Y - action
        self.quot = torch.Tensor([1])
        if Y != 0:
            self.quot = self.diff/round(Y.item(),0)
        self.sum += abs(self.quot)
        self.MAE = self.sum/self.epoch
        self.MAES.append(self.MAE.item())
        return loss

def machine_learn(age,sex,chestpain,bloodpressure,cholesterol,bloodsugar,rest_ecg,max_rate,exercise_angina,oldpeak):
    path = "MedicalNet.pt"
    with open(path, "rb") as f:
        model = pickle.load(f)
    total_output = ""
    inputs = [age,sex,chestpain,bloodpressure,cholesterol,bloodsugar,rest_ecg,max_rate,exercise_angina,oldpeak]
    actual_input = []
    binary = ""
    for i in inputs:
        binary += bin(ctypes.c_uint32.from_buffer(ctypes.c_float(i)).value)[2:]
    binary = binary.zfill(529)
    for v in binary:
        actual_input.append(int(v))
    pred = model(torch.Tensor(actual_input))
    if round(pred.item(),0) == 1:
        total_output = "The Patient Has CAD(Coronary Artery Disease)" + "\nConfidence Level:" + str(pred.item()*100) + "%"
        return total_output
    elif round(pred.item(),0) == 0:
        total_output = "The Patient Does Not Have CAD(Coronary Artery Disease)" + "\nConfidence Level:" + str((1-(pred.item()))*100) + "%"
        return total_output
def get_expl_1():
    return "Patient's Age in years (Numeric)"
def get_expl_2():
    return "Patient's Gender Male as 1 Female as 0 (Nominal)"
def get_expl_3():
    return "Type of chest pain categorized into 1 typical, 2 typical angina, 3 non-anginal pain, 4 asymptomatic"
def get_expl_4():
    return "Level of blood pressure at resting mode in mm/HG (Numerical)"
def get_expl_5():
    return "Serum cholestrol in mg/dl (Numeric)"
def get_expl_6():
    return "Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false"
def get_expl_7():
    return "result of electrocardiogram while at rest are represented in 3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy"
def get_expl_8():
    return "Maximum heart rate achieved (Numeric)"
def get_expl_9():
    return "Angina induced by exercise 0 depicting NO 1 depicting Yes (Nominal)"
def get_expl_10():
    return "Exercise induced ST-depression in comparison with the state of rest (Numeric)"
def start():
    return ""
app = gr.Interface(machine_learn, inputs=[gr.Slider(0, 125,label="Age",type="Int"), gr.Slider(0, 1,label="Sex",type="Int"),gr.Slider(0, 4,label="Chest Pain Type",type="Int"),gr.Slider(0, 300,label="Blood Pressure"),gr.Slider(0, 300,label="Cholesterol Level"),gr.Slider(0, 1,label="Fasting Blood Sugar",type="Int"),gr.Slider(0, 2,label="Rest Ecg",type="Int"),gr.Slider(0, 240,label="Maximum Heart Rate"),gr.Slider(0, 1,label="Exercise Angina"),gr.Slider(0, 10,label="Old Peak")],outputs="text")
with gr.Blocks() as app:
    box = gr.Textbox(label="Age")
    box2 = gr.Textbox(label="Sex")
with gr.Blocks() as app:
    box = gr.Textbox(label="Age")
    box2 = gr.Textbox(label="Sex")
    box3 = gr.Textbox(label = "Chest pain level")
    box4 = gr.Textbox(label="Blood Pressure")
    box5 = gr.Textbox(label="Cholesterol Level")
    box6 = gr.Textbox(label="Blood Sugar")
    box7 = gr.Textbox(label="rest ecg")
    box8 = gr.Textbox(label="Maximum heart rate")
    box9 = gr.Textbox(label="Exercise Angina")
    box10 = gr.Textbox(label="Old Peak")
    box11 = gr.Textbox(label="Prediction",values=machine_learn)
    btn = gr.Button(value="Submit")
    btn.click(machine_learn, inputs=[gr.Slider(0, 125,label="Age",step=1,type="number"), gr.Slider(0, 1,label="Sex",step=1),gr.Slider(0, 4,label="Chest Pain Type",step=1),gr.Slider(0, 300,label="Blood Pressure"),gr.Slider(0, 300,label="Cholesterol Level"),gr.Slider(0, 1,label="Fasting Blood Sugar",step=1),gr.Slider(0, 2,label="Rest Ecg",step=1),gr.Slider(0, 240,label="Maximum Heart Rate"),gr.Slider(0, 1,label="Exercise Angina",step=1),gr.Slider(0, 10,label="Old Peak")],outputs=[box11])
app.launch()