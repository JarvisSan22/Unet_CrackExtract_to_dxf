from utils import utils as ut 
import os,glob
from utils.models import unet
import datetime

COLAB=False
TRAIN=True 
#Datalocations 
if COLAB:
    DIRLOC="/content/Unet_CrackExtract_to_dxf"   #os.getcwd() #questa è la home del mio drive
else:
    DIRLOC=os.getcwd()
TRAIN_PATH=DIRLOC+"/data/DeepCrack/train"
TEST_PATH=DIRLOC+"/data/DeepCrack/test"
MODEL_SAVE=DIRLOC+"/model_saves"



def TrainRun(epochs=30,batch_size=50,
            n_classes=1,LR=1e-4,
            validation_split=0.2,
            callbacks=None,
            RUN_NOTE="TEST"
            ):
    #Get train data 
    train_X,train_y=ut.PrepTraningData(
        TRAIN_PATH+"/image",TRAIN_PATH+"/label",
        show=False,Patch=True,patch_size=256,GRAY=False,
        USEFULL=True,USEFULL_THRESH=150)

    #Define model
    N,H,W,C=train_X.shape
    imagesize=(W,H,C)
    models={"UNET":unet.model_UNET(imagesize,LR=LR)}
    model_results={}

    if not os.path.exists(MODEL_SAVE):
        os.makedirs(MODEL_SAVE)

    #Train Look for all models 
    for key,model in models.items():
        DATE=datetime.datetime.today().date().__str__()
        START_T=datetime.datetime.today()
        print(f'{"="*25} model {key} {"="*25}')
        print(START_T.__str__())

        results = model.fit(
            train_X,train_y,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1 #Print processing 
            )   

        modelname= f"{RUN_NOTE}_{key}_ep{epochs}_bs{batch_size}_{DATE}_{len(X_train)}x{H}x{W}x{C}"
        model.save(f'{MODEL_SAVE}/{modelname}.h5')
        #plot model results         
        ut.compare_TV(results,f"{MODEL_SAVE}/results",modelname,batch_size,epochs)
        
    
        END_T=datetime.datetime.today()
        print(END_T.__str__())
        print("Training time", END_T-START_T)
        model_results[key]={"model":model,
        "model_save":f'{MODEL_SAVE}/{modelname}.h5',
        "results":results,
        "runtime":END_T-START_T}

    return model_results

def main(run_type):
    if run_type.lower()=="train":
        model_results=TrainRun(epochs=30,batch_size=50,
            n_classes=1,LR=1e-4,
            validation_split=0.2,
            callbacks=None,
            RUN_NOTE="TEST"
            )
    elif run_type.lower()=="predict":
        print("To Do")
    


if __name__ =="__main__":
    main("train")