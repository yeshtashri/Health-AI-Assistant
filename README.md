# Health-AI-Assistant

## Executive Summary

Health-AI-Assistant represents a sophisticated healthcare analytics platform that leverages cutting-edge artificial intelligence and machine learning technologies to facilitate clinical decision support. The system integrates diabetes risk assessment, diabetic retinopathy detection, and intelligent clinical consultation capabilities within a unified, web-based interface.

## Demo video
[Click here to see demo](https://youtu.be/ncV85Jpn22s)

## Development Team

**Primary Developers:**
- Shavit - Lead Developer
- Yeshta Shri - Co-Developer

## System Architecture

### Core Functionality Modules

#### 1. Diabetes Risk Assessment Engine
- **Technology**: Random Forest Machine Learning Algorithm
- **Input**: Clinical parameters and patient demographics
- **Output**: Binary classification (Diabetic/Non-Diabetic) with confidence metrics
- **Features**: Real-time prediction with model validation

#### 2. Diabetic Retinopathy Detection System
- **Technology**: Deep Learning Convolutional Neural Networks
- **Input**: Retinal fundus images (standard formats: PNG, JPG, JPEG)
- **Output**: Binary classification (DR/No_DR) with diagnostic confidence
- **Features**: Automated image preprocessing, robust fallback mechanisms, cross-platform compatibility

#### 3. Clinical Consultation Intelligence Platform
- **Technology**: Retrieval-Augmented Generation (RAG) with Local Vector Database
- **Input**: Natural language medical queries
- **Output**: Contextual, evidence-based responses
- **Features**: Domain-specific knowledge retrieval, HuggingFace language model integration

#### 4. User Interface Framework
- **Technology**: Streamlit Web Application Framework
- **Features**: Responsive design, intuitive navigation, professional styling, cross-browser compatibility

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python Version**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 2GB available disk space
- **Network**: Internet connection for initial dependency download

### Development Environment
- **Package Manager**: pip 20.0+
- **Version Control**: Git 2.20+
- **Virtual Environment**: python-venv (recommended)

## Installation and Deployment

### Phase 1: Environment Preparation

1. **Repository Acquisition**
   ```bash
   git clone https://github.com/Shavitjnr/Health-AI-Assistant.git
   cd Health-AI-Assistant
   ```

2. **Virtual Environment Initialization**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Unix-based systems
   # OR
   venv\Scripts\activate     # Windows systems
   ```

### Phase 2: Dependency Management

1. **Core Dependencies Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Additional Framework Dependencies**
   ```bash
   pip install streamlit==1.28.1 fastai==2.7.19 torch==2.6.0 pandas==1.5.3 numpy==1.26.4 pillow==10.2.0 langchain==0.3.26 transformers==4.53.1 langchain_community==0.3.27 langchain_text_splitters==0.3.8 langchain_core==0.3.68 langchain_huggingface==0.3.0
   ```

### Phase 3: Application Deployment

1. **Service Initialization**
   ```bash
   streamlit run app.py
   ```

2. **Access Configuration**
   - **Local Access**: http://localhost:8501
   - **Network Access**: Configure firewall rules as necessary
   - **SSL/TLS**: Implement for production deployments

## Technical Specifications

### Dependency Matrix

| Component Category | Package | Version | Purpose |
|-------------------|---------|---------|---------|
| **Web Framework** | streamlit | 1.28.1 | User interface and application server |
| **Deep Learning** | fastai | 2.7.19 | Neural network training and inference |
| **ML Framework** | torch | 2.6.0 | PyTorch backend for deep learning |
| **Data Processing** | pandas | 1.5.3 | Data manipulation and analysis |
| **Numerical Computing** | numpy | 1.26.4 | Mathematical operations and arrays |
| **Image Processing** | pillow | 10.2.0 | Image handling and preprocessing |
| **NLP Framework** | langchain | 0.3.26 | Language model orchestration |
| **Transformer Models** | transformers | 4.53.1 | Pre-trained language models |
| **Vector Database** | langchain_community | 0.3.27 | Community integrations |
| **Text Processing** | langchain_text_splitters | 0.3.8 | Document chunking and processing |
| **Core Framework** | langchain_core | 0.3.68 | Core LangChain functionality |
| **HuggingFace Integration** | langchain_huggingface | 0.3.0 | HuggingFace model integration |

### System Architecture Diagram

```
Health-AI-Assistant/
├── app.py                    # Application entry point and UI controller
├── model_fix.py              # Model loading and fallback management
├── rag.py                    # RAG implementation and vector operations
├── requirements.txt          # Python dependency specifications
├── README.md                # System documentation
├── PIMA/                    # Diabetes prediction module
│   ├── best_rf_model.pkl    # Trained Random Forest model
│   ├── feature_columns.pkl  # Feature engineering specifications
│   └── vector_db/           # Knowledge base for diabetes domain
└── Two classes/             # Retinopathy detection module
    ├── models/              # Deep learning model artifacts
    └── vector_db/           # Knowledge base for ophthalmology
```

## Operational Procedures

### User Interface Navigation

1. **Dashboard Access**: Utilize the sidebar navigation for feature selection
2. **Diabetes Assessment**: Input clinical parameters for risk evaluation
3. **Retinopathy Analysis**: Upload retinal images for automated detection
4. **Clinical Consultation**: Submit medical queries for AI-powered responses

### System Administration

#### Performance Optimization
- **Memory Management**: Monitor RAM usage during image processing
- **CPU Utilization**: Optimize for multi-core processing capabilities
- **Storage Monitoring**: Ensure adequate disk space for model artifacts

#### Security Considerations
- **Data Privacy**: All processing occurs locally without external API calls
- **Access Control**: Implement appropriate authentication mechanisms for production
- **Audit Logging**: Enable comprehensive logging for compliance requirements

## Quality Assurance

### Model Reliability
- **Fallback Mechanisms**: Robust error handling for model loading failures
- **Validation Protocols**: Comprehensive testing of prediction accuracy
- **Performance Monitoring**: Continuous assessment of system performance

### Data Integrity
- **Input Validation**: Comprehensive validation of user inputs
- **Output Verification**: Quality checks for generated predictions
- **Error Handling**: Graceful degradation under adverse conditions

## Troubleshooting and Support

### Common Operational Issues

#### Model Loading Failures
- **Symptom**: Application fails to initialize models
- **Resolution**: Verify model file integrity and directory structure
- **Prevention**: Regular model file validation and backup procedures

#### Dependency Conflicts
- **Symptom**: Import errors or version incompatibilities
- **Resolution**: Utilize specified package versions and virtual environments
- **Prevention**: Maintain dependency version control and testing protocols

#### Performance Degradation
- **Symptom**: Slow response times or memory issues
- **Resolution**: Optimize system resources and model configurations
- **Prevention**: Regular performance monitoring and capacity planning

### Alternative Deployment Options

In the event of repository access issues, alternative deployment packages are available through our [Google Drive repository](https://drive.google.com/drive/folders/13wGcicxJhd13ZF85zVTD6v0Vg8T-ONdO?usp=drive_link).

## Development Framework

### Technology Stack
- **Frontend Framework**: Streamlit
- **Machine Learning**: FastAI, PyTorch
- **Natural Language Processing**: LangChain, Transformers
- **Data Processing**: Pandas, NumPy
- **Computer Vision**: Pillow, OpenCV (implicit)

### Development Standards
- **Code Quality**: PEP 8 compliance
- **Documentation**: Comprehensive inline documentation
- **Testing**: Unit and integration testing protocols
- **Version Control**: Git-based development workflow

## Compliance and Legal

### Educational Use
This application is designed and developed for educational and research purposes. Users are responsible for ensuring compliance with applicable regulations and institutional policies.

### Healthcare Compliance
When utilized in clinical environments, ensure adherence to:
- **HIPAA**: Health Insurance Portability and Accountability Act
- **GDPR**: General Data Protection Regulation (if applicable)
- **Local Regulations**: Jurisdiction-specific healthcare data protection laws

### Clinical Disclaimer
**IMPORTANT**: This application serves as a decision support tool and should not replace professional medical judgment. All clinical decisions must be made by qualified healthcare professionals in accordance with established medical protocols and standards of care.

## Support and Maintenance

### Technical Support
For technical assistance, system issues, or feature requests, please refer to the project documentation or contact the development team through appropriate institutional channels.

### Maintenance Schedule
- **Regular Updates**: Monthly dependency updates and security patches
- **Model Refinement**: Quarterly model performance assessments
- **Feature Enhancements**: Continuous improvement based on user feedback

---

**Document Version**: 1.0  
**Last Updated**: Current  
**Maintained By**: Development Team

