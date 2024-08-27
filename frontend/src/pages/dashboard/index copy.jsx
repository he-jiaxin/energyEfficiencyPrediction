import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Grid, Box, Typography, Button, Stack, Modal, CircularProgress, TextField } from '@mui/material';
import MainCard from 'components/MainCard';
import EnergyVisual from './EnergyEfficiencyVisual';
import ImageUploadDropzone from './ImageUploadDropzone';
import AnalyticEcommerce from 'components/cards/statistics/AnalyticEcommerce';
import FeatureImportance from './FeatureImportance';
import { retrainModel, getPredictions } from 'api/imageApi';
import ConfirmRetrainModal from './ConfirmRetrainModal';
// ==============================|| DASHBOARD - DEFAULT ||============================== //

export default function DashboardDefault() {
  const [slot, setSlot] = useState('Heatmap');
  const [baseFilename, setBaseFilename] = useState('');
  const [heatLoadPrediction, setHeatLoadPrediction] = useState(null);
  const [coolLoadPrediction, setCoolLoadPrediction] = useState(null);
  const [heatLoadFeatureImportance, setHeatLoadFeatureImportance] = useState([]);
  const [coolLoadFeatureImportance, setCoolLoadFeatureImportance] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSupportModalOpen, setIsSupportModalOpen] = useState(false);
  const [isRetrainModalOpen, setIsRetrainModalOpen] = useState(false);
  const [retrainHeatLoad, setRetrainHeatLoad] = useState('');
  const [retrainCoolLoad, setRetrainCoolLoad] = useState('');
  const [validationMessage, setValidationMessage] = useState('');
  const [retrainResults, setRetrainResults] = useState(null);

  const navigate = useNavigate();

  const handleOpenSupportModal = () => setIsSupportModalOpen(true);
  const handleCloseSupportModal = () => setIsSupportModalOpen(false);

  const handleImageUpload = async (filename, predictionData) => {
    setBaseFilename(filename);
    if (predictionData) {
      setHeatLoadPrediction(predictionData.heat_load_prediction?.[0] || null);
      setCoolLoadPrediction(predictionData.cool_load_prediction?.[0] || null);
      setHeatLoadFeatureImportance(predictionData.heat_load_feature_importance || []);
      setCoolLoadFeatureImportance(predictionData.cool_load_feature_importance || []);
    } else {
      console.error('Prediction data is undefined');
      setHeatLoadPrediction(null);
      setCoolLoadPrediction(null);
      setHeatLoadFeatureImportance([]);
      setCoolLoadFeatureImportance([]);
    }
  };

  const handleConfirmSupportNavigation = () => {
    handleCloseSupportModal();
    navigate('/support');
  };

  // Loading Modal
  const LoadingModal = () => (
    <Modal open={isLoading} disableEscapeKeyDown aria-labelledby="loading-modal-title" aria-describedby="loading-modal-description">
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 300,
          bgcolor: 'background.paper',
          boxShadow: 24,
          p: 4,
          borderRadius: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <Typography id="loading-modal-title" variant="h6" component="h2" gutterBottom>
          Processing Image
        </Typography>
        <CircularProgress size={60} thickness={4} sx={{ my: 2 }} />
        <Typography id="loading-modal-description" variant="body1" align="center" gutterBottom>
          Please wait while we analyze your image. This may take a few moments.
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center" gutterBottom>
          We're using advanced AI to calculate energy efficiency metrics.
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
          If the process is taking longer than expected, you may refresh the page or close background applications to improve performance.
        </Typography>
        {/* <Button color="error" onClick={onCancel} sx={{ mt: 2 }}>
          Cancel Process
        </Button> */}
      </Box>
    </Modal>
  );

  // Confirm Support Navigation Modal
  const ConfirmSupportNavigationModal = () => (
    <Modal open={isSupportModalOpen} onClose={handleCloseSupportModal}>
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          bgcolor: 'background.paper',
          boxShadow: 24,
          p: 4,
          borderRadius: 2,
          width: 400,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <Typography id="modal-title" variant="h6" component="h2">
          Warning
        </Typography>
        <Typography id="modal-description" sx={{ mt: 2 }}>
          Navigating to the help page will clear your session/data on this page. Do you wish to proceed?
        </Typography>
        <Box mt={2} display="flex" justifyContent="space-between" width="100%">
          <Button variant="contained" color="primary" onClick={handleConfirmSupportNavigation} sx={{ flexGrow: 1, mr: 1 }}>
            Go to Help
          </Button>
          <Button color="error" onClick={handleCloseSupportModal} sx={{ flexGrow: 1 }}>
            Cancel
          </Button>
        </Box>
      </Box>
    </Modal>
  );


  const handleRetrain = async () => {
    setIsLoading(true);
    try {
      const response = await retrainModel(baseFilename, retrainHeatLoad, retrainCoolLoad);
      if (response.success) {
        // Optionally, refresh predictions after retraining
        const updatedPredictions = await getPredictions(baseFilename);
        setHeatLoadPrediction(updatedPredictions.heatLoadPrediction);
        setCoolLoadPrediction(updatedPredictions.coolLoadPrediction);
        setHeatLoadFeatureImportance(updatedPredictions.heatLoadFeatureImportance);
        setCoolLoadFeatureImportance(updatedPredictions.coolLoadFeatureImportance);
      } else {
        console.error('Retraining failed:', response.message);
      }
    } catch (error) {
      console.error('Unexpected error during retraining:', error);
    } finally {
      setIsLoading(false);
      setIsRetrainModalOpen(false);
    }
  };

  // Handlers for retrain modal
  const handleOpenRetrainModal = () => setIsRetrainModalOpen(true);
  const handleCloseRetrainModal = () => setIsRetrainModalOpen(false);

  const handleConfirmRetrain = async () => {
    setValidationMessage('');

    const heatLoadValue = parseFloat(retrainHeatLoad);
    const coolLoadValue = parseFloat(retrainCoolLoad);
    if (isNaN(heatLoadValue) || isNaN(coolLoadValue)) {
      setValidationMessage('Please enter valid numbers for both heatload and coolload.');
      return;
    }

    try {
      // Correctly pass the baseFilename as the first argument
      const result = await retrainModel(baseFilename, heatLoadValue, coolLoadValue);
      if (result.success) {
        setRetrainResults(result.data);
        // console.log(result.message);
        // Optionally, refresh predictions after retraining
        // const updatedPredictions = await getPredictions(baseFilename);
        // setHeatLoadPrediction(updatedPredictions.heatLoadPrediction);
        // setCoolLoadPrediction(updatedPredictions.coolLoadPrediction);
        // setHeatLoadFeatureImportance(updatedPredictions.heatLoadFeatureImportance);
        // setCoolLoadFeatureImportance(updatedPredictions.coolLoadFeatureImportance);
      } else {
        setValidationMessage(result.message);
      }
    } catch (error) {
      console.error('Error in retraining:', error);
      setValidationMessage('An error occurred during retraining. Please try again.');
    }

    handleCloseRetrainModal();
  };

  return (
    <Grid container rowSpacing={4.5} columnSpacing={2.75}>
      <ConfirmRetrainModal
        isOpen={isRetrainModalOpen}
        onClose={handleCloseRetrainModal}
        onConfirm={handleConfirmRetrain}
        retrainHeatLoad={retrainHeatLoad}
        retrainCoolLoad={retrainCoolLoad}
        setRetrainHeatLoad={setRetrainHeatLoad}
        setRetrainCoolLoad={setRetrainCoolLoad}
        validationMessage={validationMessage}
        retrainResults={retrainResults}
      />
      <LoadingModal />
      <ConfirmSupportNavigationModal />
      {/* Left Column - Upload Image, Feature Importance */}
      <Grid item xs={12} md={6} lg={3} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Upload Image */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="h5" sx={{ mb: 1 }}>
            Upload Image
          </Typography>
          <MainCard sx={{ flexGrow: 1, height: '320px' }}>
            <ImageUploadDropzone onUpload={handleImageUpload} isLoading={isLoading} setIsLoading={setIsLoading} />
          </MainCard>
        </Box>

        {/* Feature Importance Overview */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, flexGrow: 1 }}>
          <Typography variant="h5" sx={{ mb: 1 }}>
            Feature Importance Overview
          </Typography>
          <MainCard sx={{ flexGrow: 1, height: '100%' }} content={false}>
            <Box sx={{ p: 3, pb: 0, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ flexGrow: 1, mt: 2 }}>
                <FeatureImportance
                  slot={slot}
                  heatLoadFeatureImportance={heatLoadFeatureImportance}
                  coolLoadFeatureImportance={coolLoadFeatureImportance}
                />
              </Box>
            </Box>
          </MainCard>
        </Box>
      </Grid>

      {/* Right Column - Energy Efficiency Visualization and Other Widgets */}
      <Grid item xs={12} md={6} lg={9} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Box sx={{ flexGrow: 1 }}>
          <EnergyVisual
            baseFilename={baseFilename}
            slot={slot}
            setSlot={setSlot}
            heatLoadFeatureImportance={heatLoadFeatureImportance}
            coolLoadFeatureImportance={coolLoadFeatureImportance}
          />
        </Box>

        {/* Row with Total Page Views, Total Users, Help & Support, Retrain Model */}
        <Grid container spacing={3} sx={{ height: 'auto' }}>
          <Grid item xs={12} sm={6} md={3}>
            <AnalyticEcommerce
              title="Heatload prediction"
              count={heatLoadPrediction !== null ? heatLoadPrediction.toFixed(2) : null}
              height="147px"
              unit="kWh/m²"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <AnalyticEcommerce
              title="Coolload prediction"
              count={coolLoadPrediction !== null ? coolLoadPrediction.toFixed(2) : null}
              height="147px"
              unit="kWh/m²"
            />
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MainCard sx={{ height: '147px' }}>
              <Stack spacing={4} sx={{ height: '100%', justifyContent: 'space-between' }}>
                <Grid container justifyContent="space-between" alignItems="center">
                  <Grid item>
                    <Stack>
                      <Typography variant="h5" noWrap>
                        Need Help?
                      </Typography>
                      <Typography variant="caption" color="secondary" noWrap>
                        Docus are available for guidance
                      </Typography>
                    </Stack>
                  </Grid>
                </Grid>
                <Box display="flex" justifyContent="center">
                  <Button
                    size="small"
                    variant="contained"
                    sx={{ textTransform: 'capitalize', width: '100%' }}
                    onClick={handleOpenSupportModal}
                  >
                    Support Docus
                  </Button>
                </Box>
              </Stack>
            </MainCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MainCard sx={{ height: '147px' }}>
              <Stack spacing={4} sx={{ height: '100%', justifyContent: 'space-between' }}>
                <Grid container justifyContent="space-between" alignItems="center">
                  <Grid item>
                    <Stack>
                      <Typography variant="h5" noWrap>
                        Retrain Model
                      </Typography>
                      <Typography variant="caption" color="secondary" noWrap>
                        Typically done within 1 minute
                      </Typography>
                    </Stack>
                  </Grid>
                </Grid>
                <Box display="flex" justifyContent="center">
                  <Button
                    size="small"
                    variant="contained"
                    sx={{ textTransform: 'capitalize', width: '100%' }}
                    onClick={handleOpenRetrainModal}
                  >
                    Retrain
                  </Button>
                </Box>
              </Stack>
            </MainCard>
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
}
