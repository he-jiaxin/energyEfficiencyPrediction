import { vi, describe, it, expect, afterEach } from 'vitest';
import axios from 'axios';
import { uploadImageAndGetResults, fetchImage, getPredictions, retrainModel, fetchSpeech } from 'api/imageApi';

vi.mock('axios');

describe('Unit tests for Image API functions', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('uploadImageAndGetResults should format the response correctly', async () => {
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });
    const mockResponse = {
      data: {
        segmentation_result: '/path/to/segmentation.png',
        prediction: {},
        heatmap: '/path/to/heatmap.png'
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    const result = await uploadImageAndGetResults(mockImageFile);

    expect(result).toEqual({
      baseFilename: 'example',
      segmentationResultPath: '/path/to/segmentation.png',
      predictionData: {},
      heatmapPath: '/path/to/heatmap.png'
    });
  });

  it('uploadImageAndGetResults should throw an error when API fails', async () => {
    const mockImageFile = new File(['dummy content'], 'example.png', { type: 'image/png' });
    axios.post.mockRejectedValueOnce(new Error('API failed'));

    await expect(uploadImageAndGetResults(mockImageFile)).rejects.toThrow('API failed');
  });

  it('fetchImage should return a URL', async () => {
    const mockBlob = new Blob(['image content'], { type: 'image/png' });
    axios.get.mockResolvedValueOnce({ data: mockBlob });

    const url = await fetchImage('example', 'segmentation');

    expect(url).toBeTruthy();
  });

  it('fetchImage should throw an error when fetching fails', async () => {
    axios.get.mockRejectedValueOnce(new Error('Fetching failed'));

    await expect(fetchImage('example', 'segmentation')).rejects.toThrow('Fetching failed');
  });

  it('getPredictions should format prediction data correctly', async () => {
    const mockResponse = {
      data: {
        cool_load_prediction: [100],
        heat_load_prediction: [200],
        heat_load_feature_importance: [],
        cool_load_feature_importance: []
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    const result = await getPredictions('example.png');

    expect(result).toEqual({
      coolLoadPrediction: 100,
      heatLoadPrediction: 200,
      heatLoadFeatureImportance: [],
      coolLoadFeatureImportance: []
    });
  });

  it('getPredictions should throw an error when no data is returned', async () => {
    axios.post.mockResolvedValueOnce({ data: null });

    await expect(getPredictions('example.png')).rejects.toThrow('No data returned from API.');
  });

  it('retrainModel should return success message when model is retrained successfully', async () => {
    const mockResponse = {
      data: {
        status: 'models retrained successfully',
        fitted_data: {}
      }
    };
    axios.post.mockResolvedValueOnce(mockResponse);

    const result = await retrainModel('example', 100, 200);

    expect(result).toEqual({
      success: true,
      message: 'Model retrained successfully',
      data: {}
    });
  });

  it('retrainModel should handle retraining failure correctly', async () => {
    axios.post.mockResolvedValueOnce({ data: { status: 'retraining failed' } });

    const result = await retrainModel('example', 100, 200);

    expect(result).toEqual({
      success: false,
      message: 'An error occurred during retraining',
      error: new Error('Retraining failed')
    });
  });

  it('fetchSpeech should return an audio URL on success', async () => {
    const mockBlob = new Blob(['audio content'], { type: 'audio/wav' });
    axios.post.mockResolvedValueOnce({ data: mockBlob });

    const url = await fetchSpeech('Hello world');

    expect(url).toBeTruthy();
  });

  it('fetchSpeech should handle rate limit errors correctly', async () => {
    axios.post.mockRejectedValueOnce({ response: { status: 429 } });

    await expect(fetchSpeech('Hello world')).rejects.toThrow('Text-to-speech API rate limit exceeded. Please try again later.');
  });
});