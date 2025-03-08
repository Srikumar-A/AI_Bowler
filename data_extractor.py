import numpy as np

def get_batsman_depth(data_df):
    '''
    Takes in player posture dataframe as input
    outputs batsman's left shoulder average depth.
    A way to find the depth the batsman is in to estimate where the ball is in batsman's reach.
    '''
    batDepth=0
    for i in range(len(data_df)):
        batDepth+=data_df.iloc[i].left_shoulder[2]
    batDepth/=len(data_df)
    return batDepth

    

def ball_state(player_posture,ball_props,frame_rate=60):
    '''
    Function to process the ball trajectory information and reward based on its trajectory.
    takes in ball props and player posture returned by the player_ball_det function.
    returns dataset
    '''
    del_x=[None,]
    del_y=[None,]
    del_z=[None,]
    del_frame=[None,]
    for i in range(0,len(ball_props)-1):
        t1=ball_props.ball_vector[i]
        t2=ball_props.ball_vector[i+1]
        del_frame+=[ball_props.frame_id[i+1]-ball_props.frame_id[i],]
        del_x+=[t2[0]-t1[0],]
        del_y+=[t2[1]-t1[1],]
        del_z+=[t2[2]-t1[2],]


    ball_props['del_x']=del_x
    ball_props['del_y']=del_y
    ball_props['del_z']=del_z
    ball_props['del_frame']=del_frame

    #removing outliers from the recorded data
    frame_list=ball_props.frame_id
    drop_list=[]
    for i in range(0,int(len(ball_props)/2)):
        if(frame_list[i+1]-frame_list[i]>=10):
            drop_list+=[frame_list[i]]
        if(frame_list[len(ball_props)-i-1]-frame_list[len(ball_props)-i-2]>=10):
            drop_list+=[frame_list[len(ball_props)-i-1]]
    ball_props=ball_props[~ball_props.frame_id.isin(drop_list)]

    #calculating velocity of the ball
    ball_props['velocity']=ball_props.del_z*frame_rate/ball_props.del_frame
    ball_props=ball_props.fillna(0)

    '''
    getting the pitching length and line, using trend analysis and gets the frame number.
    '''
    start=2
    trend=0
    length=None
    line=None
    reqd_frame=None
    for i in range(start,len(ball_props)):
        if i==start:
            if ball_props.del_y[i]>=0:
                trend=1
        elif ball_props.del_y[i]==0:
            pass
        elif (trend==1 and ball_props.del_y[i]<0) or (trend==0 and ball_props.del_y[i]>=0):
            length=ball_props.ball_vector[i-1][2]
            line=ball_props.ball_vector[i-1][0]
            reqd_frame=ball_props.frame_id[i-1]
            break
    print('ball pitching length: ',length)
    print('frame no: ',reqd_frame)
    
    '''
    Get the velocity:

    using interquartile range to remove outliers
    based on 
    upper bound: 2nd quartile+1.5*IQR
    lower bound:1st quartile-1.5*IQR
    '''
    #sorting the velocities for removal of outliers
    velocities=np.sort(ball_props.velocity.to_numpy())
    velocities=np.array([i for i in velocities if i>0])
    _1stQuart=int((len(velocities)-1)*0.25)
    _2ndQuart=int((len(velocities)-1)*0.75)
    IQR=velocities[_2ndQuart]-velocities[_1stQuart]
    velocities=np.array([i for i in velocities if i>=velocities[_1stQuart]-1.5*IQR  and i<=velocities[_2ndQuart]+1.5*IQR])
    velocity=velocities.mean()
    
    '''
    getting rewards based on the bowling and manual labelling the batsman's reaction.
    '''
    batDepth=get_batsman_depth(player_posture)
    #local variables 
    
    delDepth=99999
    ballVectorAtDepth=None
    for i in range(start,len(ball_props)):
        del_depth=batDepth-ball_props.iloc[i].ball_vector[2]
        if del_depth<delDepth and del_depth>0:
            delDepth=del_depth
            ballVectorAtDepth=ball_props.iloc[i].ball_vector

   
    #get rewards 
    #reward=get_reward(ballVectorAtDepth)
    return length,line,velocity,ballVectorAtDepth,ball_props

