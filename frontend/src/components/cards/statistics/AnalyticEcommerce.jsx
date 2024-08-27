import PropTypes from 'prop-types';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { Tooltip, Box } from '@mui/material';
import MainCard from 'components/MainCard';
import RiseOutlined from '@ant-design/icons/RiseOutlined';
import FallOutlined from '@mui/icons-material/ArrowDropDown';

const iconSX = { fontSize: '0.75rem', color: 'inherit', marginLeft: 0, marginRight: 0 };

// Function to rate energy usage based on the new bands
const rateEnergyUsage = (energy) => {
  if (energy < 11) return 6;
  if (energy < 14) return 5.5;
  if (energy < 17) return 5;
  if (energy < 20) return 4.5;
  if (energy < 23) return 4;
  if (energy < 26) return 3.5;
  if (energy < 29) return 3;
  if (energy < 32) return 2.5;
  if (energy < 35) return 2;
  if (energy < 38) return 1.5;
  return 1;
};

export default function AnalyticEcommerce({ color = 'primary', title, count, percentage, isLoss, extra, height, unit }) {
  // Default message if count is not provided or if it's 0
  const displayCount = count !== null && count !== undefined && count !== 0 ? `${count} ${unit || ''}` : 'No data available';
  const displayExtra = extra || 'No score available';

  // Calculate the energy rating based on the count value
  const energyRating = rateEnergyUsage(count);

  return (
    <MainCard sx={{ height }} contentSX={{ p: 2.25 }}>
      <Stack spacing={0.5}>
        <Typography variant="h6" color="text.secondary">
          {title}
        </Typography>
        <Grid container alignItems="center">
          <Grid item>
            <Typography variant="h4" color="inherit">
              {displayCount} {/* Display energy value */}
            </Typography>
          </Grid>
          {percentage && (
            <Grid item>
              <Chip
                variant="combined"
                color={color}
                icon={isLoss ? <FallOutlined style={iconSX} /> : <RiseOutlined style={iconSX} />}
                label={`${percentage}%`}
                sx={{ ml: 1.25, pl: 1 }}
                size="small"
              />
            </Grid>
          )}
        </Grid>
      </Stack>
      <Box sx={{ pt: 2.25 }}>
        <Typography variant="caption" color="text.secondary">
          The score of this prediction is
        </Typography>
        <Tooltip
  title={`
    Energy rating is calculated based on:
    1. Apartment size in square meters.
    2. Heating/Cooling Loads:
       - Heating: Energy needed for winter (6 months).
       - Cooling: Energy needed for summer (6 months).
    3. Combined annual energy use determines the final rating, following the NABERS standard.
  `}
  arrow
>
          <Typography
            variant="h4" // Makes the text larger
            sx={{ fontWeight: 'bold', color: `${color || 'primary'}.main`, cursor: 'pointer' }} // Makes the text bold and applies color
          >
            {count ? `${energyRating} stars` : 'No score available'}
          </Typography>
        </Tooltip>
      </Box>
    </MainCard>
  );
}

AnalyticEcommerce.propTypes = {
  color: PropTypes.string,
  title: PropTypes.string,
  count: PropTypes.oneOfType([PropTypes.string, PropTypes.number]), // Allow string or number
  percentage: PropTypes.number,
  isLoss: PropTypes.bool,
  extra: PropTypes.string,
  height: PropTypes.string, // New prop for height
  unit: PropTypes.string // New prop for unit
};