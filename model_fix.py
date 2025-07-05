import os
import torch
import torch.nn as nn
from fastai.vision.all import *
from PIL import Image
import streamlit as st

def create_simple_retinopathy_model():
    """Create a simple CNN model for retinopathy detection as fallback"""
    class SimpleRetinopathyModel(nn.Module):
        def __init__(self, num_classes=2):
            super(SimpleRetinopathyModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    return SimpleRetinopathyModel()

def load_retinopathy_model_with_fallback():
    """Load retinopathy model with multiple fallback options"""
    model_path = 'Two classes/models'
    exported_model_path = os.path.join(model_path, 'colored_export.pkl')
    
    # Try multiple approaches to load the model
    approaches = [
        ("Original model", lambda: load_learner(exported_model_path, cpu=True)),
        ("Original model with map_location", lambda: load_learner(exported_model_path, map_location='cpu')),
        ("Stage2 model", lambda: load_learner(os.path.join(model_path, 'colored_stage2.pth'), cpu=True)),
        ("Stage1 model", lambda: load_learner(os.path.join(model_path, 'colored_stage1.pth'), cpu=True))
    ]
    
    for approach_name, load_func in approaches:
        try:
            st.info(f"Trying to load model using: {approach_name}")
            model = load_func()
            st.success(f"✅ Successfully loaded model using: {approach_name}")
            return model, True
        except Exception as e:
            st.warning(f"❌ Failed to load model using {approach_name}: {str(e)}")
            continue
    
    # If all approaches fail, create a simple fallback model
    st.warning("⚠️ All model loading approaches failed. Creating a simple fallback model.")
    st.info("Note: The fallback model will not have the same accuracy as the trained model.")
    
    # Create a simple learner with the fallback model
    try:
        # Create dummy data for the learner
        path = Path('Two classes/Datasets/archive/colored_images')
        if path.exists():
            data_block = DataBlock(
                blocks=(ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(valid_pct=0.2, seed=42),
                get_y=parent_label,
                item_tfms=Resize(224),
                batch_tfms=aug_transforms()
            )
            dls = data_block.dataloaders(path, bs=1, num_workers=0)
            
            # Create learner with simple model
            simple_model = create_simple_retinopathy_model()
            learn = Learner(dls, simple_model, loss_func=nn.CrossEntropyLoss())
            
            st.info("✅ Fallback model created successfully")
            return learn, False  # False indicates it's a fallback model
        else:
            st.error("❌ Dataset path not found for fallback model creation")
            return None, False
    except Exception as e:
        st.error(f"❌ Failed to create fallback model: {str(e)}")
        return None, False

def predict_retinopathy_robust(model, image_path, is_trained_model=True):
    """Robust prediction function that handles both trained and fallback models"""
    try:
        if model is None:
            return "Model not available", None
        
        img = PILImage.create(image_path)
        
        if is_trained_model:
            # Use the trained model
            pred_class, pred_idx, probs = model.predict(img)
            return pred_class, probs
        else:
            # Use the fallback model
            with torch.no_grad():
                # Preprocess image for the simple model
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                
                # Get prediction
                output = model.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(output, dim=1).item()
                
                # Map to class names
                classes = ['No_DR', 'DR']
                pred_class = classes[pred_idx]
                
                return pred_class, probs[0].numpy()
                
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Prediction failed", None

if __name__ == "__main__":
    # Test the model loading
    model, is_trained = load_retinopathy_model_with_fallback()
    if model:
        print(f"Model loaded successfully. Is trained model: {is_trained}")
    else:
        print("Failed to load any model")

    # Export the model
    if model and is_trained:
        model.export('models/colored_export.pkl')
        print("Model exported successfully")
    else:
        print("Model not available for export") 