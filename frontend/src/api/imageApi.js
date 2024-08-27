import axios from 'axios';

const FLASK_API_URL = 'http://localhost:5001';

export const uploadImageAndGetResults = async (imageFile) => {
  const formData = new FormData();
  formData.append('input_image', imageFile);
  
  // Log the original file name
  // console.log('Original file name:', imageFile.name);
  
  // Extract base filename (without extension)
  const baseFilename = imageFile.name.replace(/\.[^/.]+$/, "");
  
  // Log the extracted base filename
  // console.log('Extracted base filename:', baseFilename);
  
  try {
    const uploadResponse = await axios.post(`${FLASK_API_URL}/segment`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    const result = uploadResponse.data;

    return {
      baseFilename,
      segmentationResultPath: result.segmentation_result,
      predictionData: result.prediction,
      heatmapPath: result.heatmap
    };
  } catch (error) {
    console.error('Error during image upload and processing:', error);
    throw error;
  }
};

export const fetchImage = async (baseFilename, type) => {
  try {
    const filename = `${baseFilename}_${type.toLowerCase()}.png`;
    const response = await axios.get(`${FLASK_API_URL}/get-image/${filename}`, {
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error('Error fetching image:', error);
    throw error;
  }
};

export const getPredictions = async (imageFileName) => {
  try {
    // Extract base filename (without extension)
    const baseFilename = imageFileName.replace(/\.[^/.]+$/, "");
    const csvPath = `/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output/${baseFilename}_calculation.csv`;

    // console.log('CSV Path:', csvPath);

    const response = await axios.post(`${FLASK_API_URL}/predict`, { csv_path: csvPath });

    if (!response.data) {
      throw new Error('No data returned from API.');
    }

    return {
      coolLoadPrediction: response.data.cool_load_prediction?.[0] || null,
      heatLoadPrediction: response.data.heat_load_prediction?.[0] || null,
      heatLoadFeatureImportance: response.data.heat_load_feature_importance || [],
      coolLoadFeatureImportance: response.data.cool_load_feature_importance || []
    };
  } catch (error) {
    console.error('Error fetching predictions:', error.message);
    throw error;
  }
};


export const retrainModel = async (baseFilename, heatload, coolload) => {
  console.log('Base filename in retrainModel:', baseFilename); // Log the received baseFilename
  
  try {
    const csvPath = `/Users/jiaxinhe/Desktop/IXN/TF2DeepFloorplan/output/${baseFilename}_calculation.csv`;

    // console.log("Sending retrain request with:", {
    //   csv_path: csvPath,
    //   heat_load: heatload,
    //   cool_load: coolload
    // });

    const response = await axios.post(`${FLASK_API_URL}/retrain`, {
      csv_path: csvPath,
      heat_load: heatload,
      cool_load: coolload
    });

    if (response.data && response.data.status === 'models retrained successfully') {
      console.log('Fitted data:', response.data.fitted_data);

      return {
        success: true,
        message: 'Model retrained successfully',
        data: response.data.fitted_data
      };
    } else {
      throw new Error('Retraining failed');
    }
  } catch (error) {
    console.error('Error during model retraining:', error);
    console.error('Server responded with:', error.response?.data);
    return {
      success: false,
      message: error.response?.data?.error || 'An error occurred during retraining',
      error: error
    };
  }
};


export async function fetchSpeech(text) {
  try {
    const response = await axios.post('http://localhost:5001/api/text-to-speech', { text }, { responseType: 'blob' });
    const audioBlob = new Blob([response.data], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);

    return audioUrl;
  } catch (error) {
    console.error('Error fetching speech:', error);

    // Customize the error handling based on status code or error type
    if (error.response) {
      // Server responded with a status other than 200 range
      if (error.response.status === 429) {
        // Handle rate limit error (too many requests)
        throw new Error('Text-to-speech API rate limit exceeded. Please try again later.');
      } else if (error.response.status === 500) {
        // Handle server error
        throw new Error('Internal server error. Please try again later.');
      } else {
        // Other specific server responses
        throw new Error(`Unexpected server response: ${error.response.statusText}`);
      }
    } else if (error.request) {
      // No response was received from the server
      throw new Error('No response from the text-to-speech service. Please check your network connection.');
    } else {
      // Something else happened in setting up the request
      throw new Error('Error occurred while making the request. Please try again.');
    }
  }
}

