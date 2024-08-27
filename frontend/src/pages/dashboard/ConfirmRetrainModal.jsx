import React, { useRef, useEffect, useState } from 'react';
import { Modal, Box, Typography, TextField, Button, Collapse, IconButton } from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

const ConfirmRetrainModal = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  retrainHeatLoad, 
  retrainCoolLoad, 
  setRetrainHeatLoad, 
  setRetrainCoolLoad, 
  validationMessage,
  retrainResults 
}) => {
  const heatloadInputRef = useRef(null);
  const [showDefinitions, setShowDefinitions] = useState(false);

  useEffect(() => {
    if (isOpen) {
      heatloadInputRef.current?.focus();
    }
  }, [isOpen]);

  const toggleDefinitions = () => {
    setShowDefinitions(!showDefinitions);
  };

  return (
    <Modal open={isOpen} onClose={onClose}>
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
          width: 500, // Increased width
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center'
        }}
      >
        <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
          Retrain Model
        </Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          To improve the accuracy of our model and ensure unbiased predictions, we ask for your input. Please provide the actual heatload
          and coolload values observed in your scenario. This helps us fine-tune the model with real-world data.
        </Typography>
        <Typography variant="body2" color="warning.main" sx={{ mb: 2 }}>
          Note: Retraining the model might affect its performance. Ensure that the provided values are accurate.
        </Typography>
        <TextField
          inputRef={heatloadInputRef}
          id="heatload-input"
          fullWidth
          label="Heatload (kWh/m²)"
          variant="outlined"
          value={retrainHeatLoad}
          onChange={(e) => setRetrainHeatLoad(e.target.value)}
          sx={{ mb: 2 }}
        />
        <TextField
          id="coolload-input"
          fullWidth
          label="Coolload (kWh/m²)"
          variant="outlined"
          value={retrainCoolLoad}
          onChange={(e) => setRetrainCoolLoad(e.target.value)}
          sx={{ mb: 2 }}
        />

        {retrainResults && (
          <Box sx={{ mt: 3, width: '100%' }}>
            <Typography variant="h6" component="h3" sx={{ mb: 2 }}>
              Retrain Results
            </Typography>
            <Typography variant="body2">Heat Load: {retrainResults.heat_load}</Typography>
            <Typography variant="body2">Cool Load: {retrainResults.cool_load}</Typography>
            <Typography variant="body2">MAE Cool: {retrainResults.mae_cool}</Typography>
            <Typography variant="body2">MAE Heat: {retrainResults.mae_heat}</Typography>
            <Typography variant="body2">MSE Cool: {retrainResults.mse_cool}</Typography>
            <Typography variant="body2">MSE Heat: {retrainResults.mse_heat}</Typography>
            <Typography variant="body2">R² Cool: {retrainResults.r2_cool}</Typography>
            <Typography variant="body2">R² Heat: {retrainResults.r2_heat}</Typography>
            <Typography variant="body2" sx={{ mt: 2 }}>Extracted Features:</Typography>
            <ul>
  {retrainResults.new_features.map((feature, index) => (
    <li key={index}>
      <Typography variant="body2">
        {`Relative Compactness: ${feature[0]}, Wall Area: ${feature[1]}, Roof Area: ${feature[2]}, Overall Height: ${feature[3]}, Orientation: ${feature[4]}, Glazing Area: ${feature[5]}, Glazing Distribution: ${feature[6]}`}
      </Typography>
    </li>
  ))}
</ul>

            {/* Collapsible Definitions Section */}
            <Box sx={{ mt: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} onClick={toggleDefinitions}>
                <Typography variant="h6" component="h4" sx={{ mb: 2 }}>
                  Definitions
                </Typography>
                <IconButton size="small" sx={{ ml: 1 }}>
                  {showDefinitions ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
              </Box>
              <Collapse in={showDefinitions}>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  <strong>MAE (Mean Absolute Error):</strong> The average of the absolute differences between predicted values and observed values. It measures the accuracy of predictions by calculating the average magnitude of the errors in a set of predictions, without considering their direction.
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  <strong>MSE (Mean Squared Error):</strong> The average of the squared differences between predicted values and observed values. It gives a higher weight to large errors, making it sensitive to outliers.
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  <strong>R² (R-squared):</strong> A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It indicates how well the model fits the data.
                </Typography>
              </Collapse>
            </Box>
          </Box>
        )}

        <Box sx={{ mt: 3, width: '100%', display: 'flex', justifyContent: 'space-between' }}>
          <Button variant="contained" color="primary" onClick={onConfirm} sx={{ flexGrow: 1, mr: 1 }}>
            Retrain
          </Button>
          <Button color="error" onClick={onClose} sx={{ flexGrow: 1 }}>
            Cancel
          </Button>
        </Box>
        {validationMessage && (
          <Typography color="error" variant="body2" sx={{ mt: 2 }}>
            {validationMessage}
          </Typography>
        )}
      </Box>
    </Modal>
  );
};

export default ConfirmRetrainModal;