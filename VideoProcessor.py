import torch
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from data_extractor import ball_state
import matplotlib.patches as patches
from PyQt5.QtCore import QThread,pyqtSignal,QMutex,QWaitCondition
import json
from Agent import AIBowler
from RLEnv import CricketEnv
import tensorflow as tf
import os

class VideoProcessor(QThread):
    pro_frame=pyqtSignal(object)
    request_input=pyqtSignal(str)
    processing_flag=pyqtSignal(str)
    preds=pyqtSignal(dict)
    end_signal=pyqtSignal()

    def __init__(self,file_path):
        super().__init__()
        self.file_path=file_path
        self.mutex=QMutex()
        self.wait_condition=QWaitCondition()
        self.isRunning_=True
        self.recieved_inp=False
        self.inp=False
        self.raw_dataset=pd.read_csv(r'dataset.csv')
        self.dataset=self.raw_dataset
        self.dataset_processor()
        # initialize model and environment
        self.env=CricketEnv(self.dataset,valid_dataset=self.dataset)
        self.Agent_B=AIBowler(env=self.env)
        if os.path.isdir(self.Agent_B.actor.chkpt_dir):
              self.Agent_B.load_models()
    def get_reward(self,ballVectorAtDepth):
        '''Setting reward system:
        things to consider
        1. Wide ball - (-500 pts)
        2. No ball(throwing to the batsman's head) (-999 pts)
        3. batsman hitting the ball(a major deflection only considered)(-100 pts) - manual(sensor gave us away)
        4. +100 if the batsman misses it.

            Change this based on performance of the models and experimentation.
                This block of code finds the trend of the ball vector to analyze and get pitching length
        '''
        # wide x axis thresholds
        wideXThresh_off=50
        wideXThresh_leg=250
        wideYThresh=168
        noballYThresh=250

        reward=0

        if ballVectorAtDepth[1]<=noballYThresh:
                print("No Ball")
                reward-=999
        elif ballVectorAtDepth[0]<=wideXThresh_off or ballVectorAtDepth[0]>=wideXThresh_leg:
                reward-=500
        '''
        Manual labelling part
        The outcome of the event previously sensor's duty, has been sized down to manual entry- sensor device wasnt stable.
        '''
        outcome=False
        self.mutex.lock()
        self.received_inp=False
        self.request_input.emit("Did the batsman hit the ball convincingly")
                
        self.wait_condition.wait(self.mutex)
        self.mutex.unlock()
        outcome=self.inp
        if outcome:
                reward-=100
        else:
                reward+=100
        return reward    
    
    def PoseToState(self,player_pos):
        state_instance=[]
        for i in range(len(player_pos)):
              state_timestamp=[]
              for k in player_pos.columns:
                    state_timestamp+=player_pos.loc[i,k]
              state_instance+=[state_timestamp]
        print(np.array(state_instance),np.array(state_instance).shape)
        return [np.array(state_instance)]
    def dataset_processor(self):
        state=[]
        for i in range(len(self.raw_dataset)):
                data=json.loads(self.raw_dataset.player_posture[i])
                state_ex=[]
                for i in range(len(data["ball_vector"])):
                      state_timestep=[]
                      for k in data.keys():
                            state_timestep+=data[k][str(i)]
                      state_ex+=[state_timestep]
                state+=[np.array([state_ex])]
        
        #adding the columns to processed dataset
        self.dataset["state"]=state
          
    def run(self):
        self.processing_flag.emit("Extracting Features")
        player_pos,ball_props=self.posture_ball_det(self.file_path)
        self.processing_flag.emit("Processing extracted features")
        length,line,velocity,ball_vector,ball_params=ball_state(player_pos,ball_props)
        player_pos=pd.merge(ball_params.drop(['del_x','del_y','del_z','del_frame','velocity'],axis=1),player_pos,on='frame_id')
        player_pos=player_pos.drop(['frame_id'],axis=1)
        reward=self.get_reward(ball_vector)

        # further processing of data to find whether some value is none
        state=self.PoseToState(player_pos)
        if velocity is None or velocity<60:
              velocity=60
        elif velocity>130:
              velocity=130
        # if none just say the video is not processable(reboot the application)
        if state is not None and length is not None and line is not None:
              # add it to dataset dataframe
              self.dataset._append({
                    'length':length,
                    'line':line,
                    'velocity':velocity,
                    'reward':reward,
                    'state':state
                    
              },ignore_index=True)
              # add the new instance to the csv dataset
              self.raw_dataset._append({
                    'player_posture':player_pos.to_json(),
                    'length':length,
                    'line':line,
                    'velocity':velocity,
                    'reward':reward
              },ignore_index=True)
              self.raw_dataset.to_csv(r'dataset.csv')

              #train RL algorithm and get the output
              self.processing_flag.emit("Predicting action")
              action=self.Agent_B.choose_action(state,evaluate=True)
              action=tf.squeeze(action)
              #return output to the frontend
              self.processing_flag.emit("Next step of action predicted")
              self.preds.emit({
                    'line':action[0].numpy(),
                    'length':action[1].numpy(),
                    'velocity':action[2].numpy()
              })
              #learn
              new_state,_,done,info=self.env.step(action)
              self.Agent_B.remember(state,action,reward,new_state,done)
              #self.processing_flag.emit("Updating the agent")
              self.Agent_B.learn()
              
              
              
              
        else:
              self.request_input.emit("The video cannot be processed as the line and length cannot be made out of the video.")

        self.end_signal.emit()



    def d_map_generator(self,frame,MiDaS,device):
        stump_detector=YOLO(r'detection_model\stump_detector.pt')
        #Detect the stump
        stump_preds=stump_detector(frame)
        # show the detection to the user for validation
        stump_base=[]
        fig,ax=plt.subplots(1)
        ax.imshow(frame)
        ax.set_axis_off()
        for res in stump_preds:
                boxes=res.boxes.cpu().numpy()
                xyxys=boxes.xyxy
                for xyxy in xyxys:
                        print(xyxy)
                        rect=patches.Rectangle((xyxy[0],xyxy[1]),xyxy[2]-xyxy[0],xyxy[3]-xyxy[1],linewidth=1,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
                        stump_base+=[[(xyxy[0]+xyxy[2])/2,xyxy[3]],]
        self.pro_frame.emit(fig)
        global user_validation  
        user_validation=False
        if len(stump_base)!=0:
                self.mutex.lock()
                self.received_inp=False
                self.request_input.emit("If the stump is detected press yes else no, \n Note:Only one stump set should be there and detected")
                
                self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
        else:
             self.inp=False
        #user_validation=input('Enter 0 for no and 1 for yes : ')
        #user_validation=1
    
  
    

        '''
        Generating D-Map to get 3D from 2D images
        '''
        user_validation=self.inp
        if user_validation:
                #loading dpt transforms 
                print("---inside d_map gen 2",frame.shape)    
                midas_transforms=torch.hub.load('intel-isl/MiDaS','transforms')
                transform=midas_transforms.dpt_transform
                #preprocess the frame
                frame_=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    
                input_frame=transform(frame_).to(device) 
       
    


                #d-map generation
                with torch.no_grad():
                        d_map=MiDaS(input_frame)
                        d_map=torch.nn.functional.interpolate(
                        d_map.unsqueeze(1),
                        size=input_frame.shape[2:],
                        mode='bicubic',
                        align_corners=False,
                                ).squeeze()
                        output=d_map.cpu().numpy()
                # getting the scale for the frame
                '''
                logic for the scaling, if one stump is detected then the base of the image is taken as the
                end of the pitch and the pitch is rescaled to 22 yards or 20.1168 meters.
                if two stumps are detected then the distance between their bases is considered.
                if more than 2 or no stumps are detected, the loop is makes it consider the next frame.
                '''
                height,width=input_frame.shape[:2]
                scale=12/(d_map[int(stump_base[0][1])][int(stump_base[0][0])]-
                    d_map[int(height)-1][int(stump_base[0][0])])
                valid_flag=True
    
                print("stump coords",stump_base)
                '''
                Remove the second condition of two stumps in the frame
                '''
                if len(stump_base)==1 or len(stump_base)==2:
                        pass
                else:
                      valid_flag=False

                if valid_flag:
                        return output*scale.item(),True,stump_base
                else:
                        return None,False,[]
        return None,False,[]
    def posture_ball_det(self,vid_loc,orientation='right'):
        '''
        Takes in two arguments vid_loc raw string that points to the location the video to be processed is stored and orientation which corressponds to right/left handed batsman string.
        Function that uses d_map_generator function to get the real coordinates of the player and ball from each frame.
        '''
        frame_id=0
        scale=0
        #loading the model
        MiDaS=torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
        #setting up the device gpu
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        MiDaS.to(device)
        MiDaS.eval()

        mp_pose=mp.solutions.pose
        mp_drawing=mp.solutions.drawing_utils

        cap=cv2.VideoCapture(vid_loc)
        # the number of frames may range from 200-400 max
        '''
        ball_props
        '''
        ball_props=pd.DataFrame(columns=['ball_vector','frame_id'])
        player_posture=pd.DataFrame(columns=['frame_id','nose','left_eye',
                                        'right_eye','left_shoulder',
                                        'right_shoulder','left_elbow',
                                        'right_elbow','left_wrist',
                                        'right_wrist','left_hip',
                                        'right_hip','left_knee',
                                        'right_knee','left_ankle',
                                        'right_ankle','left_foot_index',
                                        'right_foot_index'])
        d_map_flag=False
        while cap.isOpened():
            ret,frame=cap.read()
            frame=cv2.flip(frame,0) if orientation=='right' else frame
            frame_id+=1
            
        
            if not ret:
                break
            frame=np.transpose(frame,(1,0,2))
            print(frame.shape)

            '''
            Image padding to avoid using DPT transforms that changes the image's aspect ratio
            '''
            height,width, _ = frame.shape
            target_height = 672  # Change if needed
            scale = target_height / height
            target_width = int(width * scale)

            # Resize while maintaining the aspect ratio
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)  
            # Calculate padding to make dimensions divisible by 32 (for the DPT model)
            pad_height = (32 - height % 32) % 32
            pad_width = (32 - width % 32) % 32

            # Pad the image while keeping it centered
            padded_image = cv2.copyMakeBorder(
                frame,
                top=pad_height // 2,
                bottom=pad_height - pad_height // 2,
                left=pad_width // 2,
                right=pad_width - pad_width // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],  # Black padding
            )  
            if not (d_map_flag):     
                d_map,d_map_flag,stump_base=self.d_map_generator(padded_image,MiDaS,device)   
            # get in the stump detector code and make sure once the stump is detected, the code should not get into the stump detector part again.
            if d_map_flag:
                
                '''
                Ball detection algorithm....
                The model has been developed using yolov8s image augmentation and planning to use hyperparameters tuning.
                '''
                
                ball_det_model=YOLO(r'detection_model\ball_detector_aug.pt')
                ball_preds=ball_det_model(frame)

                '''
                Posture detection algorithm....
                Have to get the model ready
                '''
                fig,ax=plt.subplots(1)
                ax.set_axis_off()
                ax.imshow(frame)
                with mp_pose.Pose() as pose:
                    pose_coords=pose.process(frame)
                    circle_radius=int(.007*frame.shape[0])
                    point_spec=mp_drawing.DrawingSpec(color=(0,0,255),thickness=-1,circle_radius=circle_radius)
                    line_spec=mp_drawing.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2)
                    mp_drawing.draw_landmarks(frame,
                                            landmark_list=pose_coords.pose_landmarks,
                                            landmark_drawing_spec=point_spec,
                                            connection_drawing_spec=line_spec
                                            )

                '''
                Storing the ball coordinates and getting its z-component.
                '''
            
                for res in ball_preds:
                    boxes=res.boxes.cpu().numpy()
                    xyxys=boxes.xyxy
                    for xyxy in xyxys:
                        print(xyxy)
                        cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),thickness=1,color=(0,255,0))
                        x=int((xyxy[0]+xyxy[2])/2)
                        y=int((xyxy[1]+xyxy[3])/2)
                        df={'frame_id':frame_id,
                            'ball_vector':[x,y,d_map[y][x]],
                            }
                        ball_props=ball_props._append(df,ignore_index=True)
                
                #plt.show()
                self.pro_frame.emit(fig)
            
                
                if len(ball_props)!=0:
                    if ball_props.loc[len(ball_props)-1].frame_id!=frame_id:
                        df={'ball_vector':[None,None,None],
                            'frame_id':frame_id}
                        ball_props._append(df,ignore_index=True)
                    
                #add posture info to the dataframe
                '''
                Doesnt take into account of frames before the ball is detected.
                '''
                if pose_coords.pose_landmarks is not None and len(ball_props)!=0:
                    df_pose={
                        'frame_id':frame_id,
                        'nose':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y*d_map.shape[1])])],
                        'left_eye':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y*d_map.shape[1])])],
                        'right_eye':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x*frame.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y*d_map.shape[1])])],
                        'left_shoulder':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y*d_map.shape[1])])],
                        'right_shoulder':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*d_map.shape[1])])],
                        'left_elbow':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y*d_map.shape[1])])],
                        'right_elbow':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y*d_map.shape[1])])],
                        'left_wrist':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*d_map.shape[1])])],
                        'right_wrist':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*frame.shape[0]),
                                    int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*d_map.shape[1])])],
                        'left_hip':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*d_map.shape[1])])],
                        'right_hip':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*d_map.shape[1])])],
                        'left_knee':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*d_map.shape[1])])],
                        'right_knee':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y*d_map.shape[1])])],
                        'left_ankle':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*d_map.shape[1])])],
                        'right_ankle':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y*d_map.shape[1])])],
                        'left_foot_index':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y*d_map.shape[1])])],
                        'right_foot_index':[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x*frame.shape[0]),
                                int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y*frame.shape[1]),
                                int(d_map[int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x*d_map.shape[0])][int(pose_coords.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y*d_map.shape[1])])],
                    }
                    player_posture=player_posture._append(df_pose,ignore_index=True)
                    
                    
            
            
            
            
        cap.release()
        cv2.destroyAllWindows()
        return player_posture,ball_props
    
    def user_response(self,inp):
         self.mutex.lock()
         self.received_inp=True
         self.inp=inp
         self.wait_condition.wakeAll()
         self.mutex.unlock()

    def stop(self):
        self.isRunning_=False