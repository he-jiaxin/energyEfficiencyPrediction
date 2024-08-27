import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Button, Stack } from '@mui/material';
import { styled } from '@mui/material/styles';
import { uploadImageAndGetResults } from 'api/imageApi'; // Adjust the import path

const DropzoneWrapper = styled('div')(({ theme }) => ({
  width: '100%',
  height: 210,
  border: `2px dashed ${theme.palette.primary.main}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  borderRadius: theme.shape.borderRadius,
  cursor: 'pointer',
  backgroundColor: theme.palette.background.default,
  overflow: 'hidden',
  position: 'relative',
  '&:hover': {
    backgroundColor: theme.palette.action.hover
  }
}));

const ImagePreview = styled('img')({
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  position: 'absolute',
  top: 0,
  left: 0
});

const PDFPreview = styled('iframe')({
  width: '100%',
  height: '100%',
  position: 'absolute',
  top: 0,
  left: 0,
  border: 'none'
});

const ImageUploadDropzone = ({ onUpload, isLoading, setIsLoading }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isPDF, setIsPDF] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false); // Loading state

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setFileName(file.name);
    setIsPDF(file.type === 'application/pdf');
    setUploadedFile(file);
  };

  const handleCancel = () => {
    setUploadedFile(null);
    setFileName('');
    setIsPDF(false);
    setPredictions(null);
    setLoading(false); // Reset loading state on cancel
  };

  const handleSubmit = async () => {
    if (!uploadedFile) return;
  
    setIsLoading(true);
  
    try {
      const uploadResponse = await uploadImageAndGetResults(uploadedFile);
  
      // Pass relevant data to the parent component
      if (onUpload) {
        onUpload(uploadResponse.baseFilename, uploadResponse.predictionData);
      }
    } catch (error) {
      console.error('Error during image upload and prediction:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
  accept: {
    'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
    'application/pdf': ['.pdf']
  },
  onDrop,
  multiple: false
});

  return (
    <Box sx={{ width: '100%', maxWidth: '100%', margin: 'auto', textAlign: 'center' }}>
      <DropzoneWrapper {...getRootProps()}>
        <input {...getInputProps()} />
        {uploadedFile ? (
          isPDF ? (
            <PDFPreview src={URL.createObjectURL(uploadedFile)} title="PDF Preview" />
          ) : (
            <ImagePreview src={URL.createObjectURL(uploadedFile)} alt="Uploaded Image Preview" />
          )
        ) : (
          <Typography variant="body2" style={{ textAlign: 'center', width: '180px', margin: '0 auto' }}>
            Drag & drop an image or PDF here, or click to select one
          </Typography>
        )}
      </DropzoneWrapper>

      {fileName && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          {fileName}
        </Typography>
      )}

      {predictions && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="h6">Predictions</Typography>
          <Typography variant="body2">Heatload: {predictions.heat_load_prediction[0]}</Typography>
          <Typography variant="body2">Coolload: {predictions.cool_load_prediction[0]}</Typography>
        </Box>
      )}

      <Stack direction="row" spacing={2} justifyContent="flex-end" alignItems="center" sx={{ mt: 2 }}>
        {/* {loading && (
          <CircularProgress size={24} /> // Adjust size if needed
        )} */}
        <Button
          color="error"
          onClick={handleCancel}
          disabled={loading} // Disable the cancel button while loading
        >
          Remove
        </Button>

        <Button
          type="submit"
          variant="contained"
          onClick={handleSubmit}
          disabled={loading} // Disable the submit button while loading
        >
          {isLoading ? 'Submitting...' : 'Submit'} {/* Button text changes based on loading state */}
        </Button>
      </Stack>
    </Box>
  );
};

export default ImageUploadDropzone;
