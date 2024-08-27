import { vi, describe, it, expect, afterEach } from 'vitest';
import axios from 'axios';
import { uploadImageAndGetResults, fetchImage, getPredictions, retrainModel, fetchSpeech } from 'api/imageApi';

vi.mock('axios');

describe('Integration tests for Image API functions', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should upload an image and fetch the segmentation result', async () => {
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });
    const mockUploadResponse = {
      data: {
        segmentation_result: '/path/to/segmentation.png',
        prediction: {},
        heatmap: '/path/to/heatmap.png'
      }
    };
    const mockFetchBlob = new Blob(['image content'], { type: 'image/png' });

    axios.post.mockResolvedValueOnce(mockUploadResponse);
    axios.get.mockResolvedValueOnce({ data: mockFetchBlob });

    const result = await uploadImageAndGetResults(mockImageFile);
    const imageUrl = await fetchImage(result.baseFilename, 'segmentation');

    expect(result.segmentationResultPath).toEqual('/path/to/segmentation.png');
    expect(imageUrl).toBeTruthy();
  });

  it('should upload an image and get predictions', async () => {
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });
    const mockUploadResponse = {
      data: {
        segmentation_result: '/path/to/segmentation.png',
        prediction: {},
        heatmap: '/path/to/heatmap.png'
      }
    };
    const mockPredictionResponse = {
      data: {
        cool_load_prediction: [100],
        heat_load_prediction: [200],
        heat_load_feature_importance: [],
        cool_load_feature_importance: []
      }
    };

    axios.post.mockResolvedValueOnce(mockUploadResponse);
    axios.post.mockResolvedValueOnce(mockPredictionResponse);

    const result = await uploadImageAndGetResults(mockImageFile);
    const predictions = await getPredictions('example.png');

    expect(predictions.coolLoadPrediction).toEqual(100);
    expect(predictions.heatLoadPrediction).toEqual(200);
  });

  it('should retrain the model and get the retraining success message', async () => {
    const mockRetrainResponse = {
      data: {
        status: 'models retrained successfully',
        fitted_data: {}
      }
    };

    axios.post.mockResolvedValueOnce(mockRetrainResponse);

    const result = await retrainModel('example', 100, 200);

    expect(result.success).toBe(true);
    expect(result.message).toEqual('Model retrained successfully');
  });

  it('should fetch speech and return an audio URL', async () => {
    const mockBlob = new Blob(['audio content'], { type: 'audio/wav' });
    axios.post.mockResolvedValueOnce({ data: mockBlob });

    const audioUrl = await fetchSpeech('Test speech');

    expect(audioUrl).toBeTruthy();
  });

  it('should handle a full workflow from image upload to fetching prediction and retraining', async () => {
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });
    const mockUploadResponse = {
      data: {
        segmentation_result: '/path/to/segmentation.png',
        prediction: {},
        heatmap: '/path/to/heatmap.png'
      }
    };
    const mockPredictionResponse = {
      data: {
        cool_load_prediction: [100],
        heat_load_prediction: [200],
        heat_load_feature_importance: [],
        cool_load_feature_importance: []
      }
    };
    const mockRetrainResponse = {
      data: {
        status: 'models retrained successfully',
        fitted_data: {}
      }
    };

    axios.post.mockResolvedValueOnce(mockUploadResponse);
    axios.post.mockResolvedValueOnce(mockPredictionResponse);
    axios.post.mockResolvedValueOnce(mockRetrainResponse);

    const uploadResult = await uploadImageAndGetResults(mockImageFile);
    const predictions = await getPredictions('example.png');
    const retrainResult = await retrainModel('example', 200, 100);

    expect(uploadResult.segmentationResultPath).toEqual('/path/to/segmentation.png');
    expect(predictions.coolLoadPrediction).toEqual(100);
    expect(retrainResult.success).toBe(true);
  });

  it('should handle errors during the full workflow', async () => {
    axios.post.mockRejectedValueOnce(new Error('Upload failed'));
    
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });

    await expect(uploadImageAndGetResults(mockImageFile)).rejects.toThrow('Upload failed');
  });

  it('should handle missing prediction data correctly', async () => {
    const mockPredictionResponse = { data: null };

    axios.post.mockResolvedValueOnce(mockPredictionResponse);

    await expect(getPredictions('example.png')).rejects.toThrow('No data returned from API.');
  });

  it('should process and return speech audio correctly', async () => {
    const mockBlob = new Blob(['audio content'], { type: 'audio/wav' });
    axios.post.mockResolvedValueOnce({ data: mockBlob });

    const audioUrl = await fetchSpeech('Hello world');

    expect(audioUrl).toBeTruthy();
  });

  it('should fail to fetch speech due to rate limit and return appropriate error', async () => {
    axios.post.mockRejectedValueOnce({ response: { status: 429 } });

    await expect(fetchSpeech('Hello world')).rejects.toThrow('Text-to-speech API rate limit exceeded. Please try again later.');
  });

  it('should handle image fetch and return a valid URL', async () => {
    const mockBlob = new Blob(['image content'], { type: 'image/png' });
    axios.get.mockResolvedValueOnce({ data: mockBlob });

    const imageUrl = await fetchImage('example', 'heatmap');

    expect(imageUrl).toBeTruthy();
  });
});