import { useState, useEffect, useRef } from 'react';
import { Button, Grid, Stack, Typography, Box, Modal, Fade, Backdrop } from '@mui/material';
import MainCard from 'components/MainCard';
import { fetchImage, getPredictions, fetchSpeech } from 'api/imageApi';
import SaveIcon from '@mui/icons-material/Save';

export async function playRecommendationsSequentially(recommendations, setIsPlaying, setIsPaused, audioRef) {
  if (audioRef.current) {
    if (audioRef.current.paused) {
      audioRef.current.play();
      setIsPaused(false);
    } else {
      audioRef.current.pause();
      setIsPaused(true);
    }
    return;
  }

  setIsPlaying(true);

  const introMessage = "These recommendations are based on the model's performance. Please be aware that they might not be 100% accurate.";
  try {
    const introAudioUrl = await fetchSpeech(introMessage);
    const introAudio = new Audio(introAudioUrl);
    audioRef.current = introAudio;
    await new Promise((resolve, reject) => {
      introAudio.onended = resolve;
      introAudio.onerror = reject;
      introAudio.play();
    });
  } catch (error) {
    console.error('Error playing introductory message:', error);
  }

  for (const rec of recommendations) {
    try {
      const audioUrl = await fetchSpeech(rec);
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      await new Promise((resolve, reject) => {
        audio.onended = resolve;
        audio.onerror = reject;
        audio.play();
      });
    } catch (error) {
      console.error('Error playing recommendation:', error);
    }
  }

  setIsPlaying(false);
  audioRef.current = null;
}

export default function EnergyEfficiencyVisual({ baseFilename, setSlot }) {
  const [imageUrl, setImageUrl] = useState(null);
  const [slot, setSlotState] = useState('Heatmap');
  const [recommendations, setRecommendations] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isApiConnected, setIsApiConnected] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false); // Modal state
  const audioRef = useRef(null);

  useEffect(() => {
    const loadImage = async () => {
      try {
        if (baseFilename) {
          const type = slot.toLowerCase();
          const url = await fetchImage(baseFilename, type);
          setImageUrl(url);
  
          const predictionData = await getPredictions(baseFilename);
  
          if (predictionData.heatLoadPrediction !== null || predictionData.coolLoadPrediction !== null) {
            const recommendations = generateRecommendations(predictionData);
            setRecommendations(recommendations);
  
            // Remove the immediate API connection check
            // setIsApiConnected(true);
          } else {
            console.warn('Received null predictions. Skipping recommendations.');
          }
        }
      } catch (error) {
        console.error('Error fetching image or predictions:', error);
        setImageUrl(null);
      }
    };
    loadImage();
  }, [baseFilename, slot]);
  

  const handleSlotChange = (newSlot) => {
    setSlotState(newSlot);
    setSlot(newSlot);
  };

  const handleSaveImage = () => {
    if (imageUrl) {
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `${slot}_${baseFilename}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const handlePlayClick = async () => {
    try {
      // Check API connection here
      await fetchSpeech("Test connection");
      setIsApiConnected(true);
      playRecommendationsSequentially(recommendations, setIsPlaying, setIsPaused, audioRef);
    } catch (error) {
      console.error('Error connecting to text-to-speech API:', error);
      setIsApiConnected(false);
      setIsModalOpen(true);
    }
  };
  const handleCloseModal = () => {
    setIsModalOpen(false); // Close modal
  };

  const generateRecommendations = (data) => {
    const { heatLoadFeatureImportance, coolLoadFeatureImportance } = data;

    const recommendations = [];

    const getFeatureImportance = (featureArray, featureName) => {
      const featureObj = featureArray.find((item) => item.Feature === featureName);
      return featureObj ? featureObj.Importance : null;
    };

    const addRecommendation = (importance, highMsg, mediumMsg, lowMsg) => {
      if (importance > 0.4) {
        recommendations.push(highMsg);
      } else if (importance > 0.2) {
        recommendations.push(mediumMsg);
      } else if (importance > 0.1) {
        recommendations.push(lowMsg);
      }
    };

    // Heat Load Recommendations
    const heatHeightImportance = getFeatureImportance(heatLoadFeatureImportance, 'overall_height');
    const heatRoofAreaImportance = getFeatureImportance(heatLoadFeatureImportance, 'roof_area');
    const heatCompactnessImportance = getFeatureImportance(heatLoadFeatureImportance, 'relative_compactness');

    addRecommendation(
      heatHeightImportance,
      'The drawing indicates that high ceilings are significantly increasing heating demands. Consider lowering ceiling heights where feasible or installing thermal barriers to improve efficiency.',
      'The drawing suggests that ceiling height moderately affects heating efficiency. Evaluate options like upgrading insulation or adding thermal curtains.',
      'According to the drawing, ceiling height has a minor impact on heating. Focus on other areas, such as windows or doors, for better heat retention.'
    );

    addRecommendation(
      heatRoofAreaImportance,
      'The drawing shows that a large roof area is contributing to heat loss. Enhancing insulation or applying reflective roof coatings could yield significant energy savings.',
      'Based on the drawing, the roof area moderately impacts heating. Ensure insulation is effective and consider checking for leaks.',
      'The drawing reveals that roof area has a smaller effect on heating. Maintain existing insulation and inspect for drafts.'
    );

    addRecommendation(
      heatCompactnessImportance,
      "The drawing highlights that the building's layout is causing significant heat loss. Consider redesigning to minimize exposed surfaces and improve heat retention.",
      'The drawing suggests that the building layout moderately affects heating efficiency. Review spaces for potential reconfiguration or add internal partitions.',
      'According to the drawing, layout has a minor impact on heating. Focus on enhancing window insulation and sealing any gaps.'
    );

    // Cool Load Recommendations
    const coolRoofAreaImportance = getFeatureImportance(coolLoadFeatureImportance, 'roof_area');
    const coolHeightImportance = getFeatureImportance(coolLoadFeatureImportance, 'overall_height');
    const coolCompactnessImportance = getFeatureImportance(coolLoadFeatureImportance, 'relative_compactness');

    addRecommendation(
      coolRoofAreaImportance,
      'The drawing indicates that the roof is absorbing significant heat, increasing cooling demands. Consider installing cool roof coatings or adding rooftop greenery.',
      'Based on the drawing, the roof area moderately affects cooling. Improve attic ventilation and consider using reflective roofing materials.',
      'The drawing shows that roof area has a smaller impact on cooling. Ensure roofing materials are in good condition and maintain proper ventilation.'
    );

    addRecommendation(
      coolHeightImportance,
      'The drawing suggests that high ceilings are trapping warm air, raising cooling demands. Install ceiling fans or ventilation systems to improve air circulation.',
      'According to the drawing, ceiling height moderately affects cooling. Consider adding ceiling fans or optimizing natural ventilation.',
      'The drawing shows that ceiling height has a minor impact on cooling. Focus on improving window shading and sealing.'
    );

    addRecommendation(
      coolCompactnessImportance,
      'The drawing reveals that the building layout is increasing cooling demands. Explore design changes to reduce exposed surface areas and enhance efficiency.',
      'The drawing suggests that compactness moderately influences cooling. Consider adding external shading devices or reflective materials to walls.',
      'The drawing shows that compactness has a smaller effect on cooling. Concentrate on improving window insulation and using lighter colors for walls.'
    );

    return recommendations;
  };

  return (
    <>
      <Grid container alignItems="center" justifyContent="space-between">
        <Grid item>
          <Typography variant="h5">Energy Efficiency Visualisation</Typography>
        </Grid>
        <Grid item>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Button
              size="small"
              onClick={() => handleSlotChange('Heatmap')}
              color={slot === 'Heatmap' ? 'primary' : 'secondary'}
              variant={slot === 'Heatmap' ? 'outlined' : 'text'}
            >
              Heatmap
            </Button>
            <Button
              size="small"
              onClick={() => handleSlotChange('Coolmap')}
              color={slot === 'Coolmap' ? 'primary' : 'secondary'}
              variant={slot === 'Coolmap' ? 'outlined' : 'text'}
            >
              Coolmap
            </Button>
            {imageUrl && (
              <Button size="small" onClick={handleSaveImage} startIcon={<SaveIcon />} variant="contained">
                Save Image
              </Button>
            )}
          </Stack>
        </Grid>
      </Grid>
      <MainCard content={false} sx={{ mt: 1.5, p: 2 }}>
        <Box
          sx={{
            minHeight: '600px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: imageUrl ? 'flex-start' : 'center',
            textAlign: 'left',
            pl: imageUrl ? 2 : 0,
            pr: 2
          }}
        >
          {imageUrl ? (
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={4.5}>
                {imageUrl && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      <strong>{slot === 'Heatmap' ? 'Heat Load' : 'Cool Load'}:</strong> The amount of{' '}
                      {slot === 'Heatmap' ? 'heat' : 'cooling'} energy required to maintain a comfortable indoor environment.
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      • Higher values {slot === 'Heatmap' ? '(red)' : '(blue)'} indicate areas needing more{' '}
                      {slot === 'Heatmap' ? 'heating' : 'cooling'}.
                    </Typography>
                    <Typography variant="body2">
                      • Lower values {slot === 'Heatmap' ? '(blue)' : '(red)'} suggest less {slot === 'Heatmap' ? 'heating' : 'cooling'} is
                      required.
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 2 }}>
                      This visualization helps in identifying areas requiring more or less {slot === 'Heatmap' ? 'heating' : 'cooling'},
                      allowing for better energy efficiency management.
                    </Typography>
                  </Box>
                )}
                {recommendations && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Recommendations:
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      These recommendations are based on the model's performance. Please be aware that they might not be 100% accurate.
                    </Typography>
                    <ul>
                      {recommendations.map((rec, index) => (
                        <li key={index}>
                          <Typography variant="body2">{rec}</Typography>
                        </li>
                      ))}
                    </ul>
                    <Button size="small" variant="contained" onClick={handlePlayClick} sx={{ mt: 2 }}>
                      {isPlaying ? (isPaused ? 'Resume' : 'Pause') : 'Play Recommendations'}
                    </Button>
                  </Box>
                )}
              </Grid>
              <Grid item xs={0.5}>
                <Box
                  sx={{
                    height: '100%',
                    borderRight: '1px solid #B0BEC5',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                />
              </Grid>
              <Grid item xs={7}>
                <Box
                  sx={{
                    pt: 1,
                    pr: 2,
                    minHeight: '600px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: imageUrl ? null : '#f5f5f5',
                    textAlign: 'left'
                  }}
                >
                  <img src={imageUrl} alt={`${slot} display`} style={{ maxWidth: '95%', maxHeight: '100%', objectFit: 'contain' }} />
                </Box>
              </Grid>
            </Grid>
          ) : (
            <Typography variant="body2">
              This section displays your Heat Map and Cool Map, providing a visual representation of temperature distribution and cooling
              efficiency across your space.
            </Typography>
          )}
        </Box>
      </MainCard>

      {/* Modal Component for Error Handling */}
      <Modal
        open={isModalOpen}
        onClose={handleCloseModal}
        closeAfterTransition
        BackdropComponent={Backdrop}
        BackdropProps={{
          timeout: 500
        }}
      >
        <Fade in={isModalOpen}>
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 400,
              bgcolor: 'background.paper',
              borderRadius: 2, // Rounded corners
              boxShadow: 24,
              p: 4,
              textAlign: 'center' // Center-align the text
            }}
          >
            <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
              Text-to-Speech Error
            </Typography>
            <Typography sx={{ mt: 2, mb: 4 }}>
              There was an error connecting to the text-to-speech API. Please check your API key and the API URL, and try again.
            </Typography>
            <Button
              onClick={handleCloseModal}
              color="error"
              sx={{ mr: 2 }} // Styling similar to the "Cancel" button
            >
              Close
            </Button>
          </Box>
        </Fade>
      </Modal>
    </>
  );
}
