import { useEffect, useState } from 'react';
import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import ReactApexChart from 'react-apexcharts';

const defaultCategories = [
  'roof_area',
  'relative_compactness',
  'overall_height',
  'glazing_area',
  'wall_area',
  'glazing_area_distribution',
  'orientation'
];

const defaultBarChartOptions = {
  chart: {
    type: 'bar',
    height: 340,
    width: '100%',
    toolbar: {
      show: false
    }
  },
  plotOptions: {
    bar: {
      columnWidth: '45%',
      borderRadius: 4
    }
  },
  dataLabels: {
    enabled: false
  },
  xaxis: {
    categories: defaultCategories,
    axisBorder: {
      show: false
    },
    axisTicks: {
      show: false
    }
  },
  yaxis: {
    min: 0,
    max: 50,
    tickAmount: 5,
    labels: {
      formatter: function (value) {
        return value.toFixed(1) + '%';
      }
    }
  },
  tooltip: {
    y: {
      formatter: function (value) {
        return value.toFixed(1) + '%';
      }
    }
  },
  grid: {
    show: false
  }
};

export default function FeatureImportance({ slot, heatLoadFeatureImportance, coolLoadFeatureImportance }) {
  const theme = useTheme();
  const { secondary } = theme.palette.text;
  const orange = theme.palette.warning.main;
  const blue = theme.palette.primary.main;

  const [series, setSeries] = useState([{ name: 'Feature Importance', data: Array(defaultCategories.length).fill(0) }]);
  const [options, setOptions] = useState(defaultBarChartOptions);

  useEffect(() => {
    const featureImportance = slot === 'Heatmap' ? heatLoadFeatureImportance : coolLoadFeatureImportance;
    const color = slot === 'Heatmap' ? orange : blue;
    const seriesName = slot === 'Heatmap' ? 'Heatload' : 'Coolload';

    if (featureImportance && featureImportance.length > 0) {
      const totalImportance = featureImportance.reduce((sum, item) => sum + item.Importance, 0);
      const data = featureImportance.map(item => parseFloat((item.Importance / totalImportance * 100).toFixed(1)));
      const categories = featureImportance.map(item => item.Feature);
      
      setSeries([{ name: seriesName, data }]);
      setOptions(prevState => ({
        ...prevState,
        colors: [color],
        xaxis: {
          ...prevState.xaxis,
          categories,
        }
      }));
    } else {
      setSeries([{ name: seriesName, data: Array(defaultCategories.length).fill(0) }]);
      setOptions(prevState => ({
        ...prevState,
        colors: [color],
        xaxis: {
          ...prevState.xaxis,
          categories: defaultCategories,
        }
      }));
    }
  }, [slot, heatLoadFeatureImportance, coolLoadFeatureImportance, secondary, orange, blue]);

  return (
    <Box id="chart" sx={{ bgcolor: 'transparent', width: '100%', height: '100%' }}>
      <ReactApexChart options={options} series={series} type="bar" height="100%" />
    </Box>
  );
}