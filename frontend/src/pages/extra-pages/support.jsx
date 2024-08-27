import Typography from '@mui/material/Typography';
import MainCard from 'components/MainCard';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';

export default function SupportPage() {
  return (
    <MainCard title="Support">
      <Typography variant="body1" gutterBottom>
        Welcome to the Support page for our Energy Efficiency Assessment Tool.
      </Typography>

      <Typography variant="body2" paragraph>
        This project is part of the UCL IXN initiative, focusing on developing an AI application that analyzes 2D architectural blueprints to predict and enhance energy efficiency. Below you will find detailed information about the key features, integration process, and available support resources for our tool.
      </Typography>

      <Typography variant="h6" gutterBottom>
        Key Features of Our Application
      </Typography>

      <List>
        <ListItem>
          <ListItemText
            primary="Data Extraction"
            secondary="Our application integrates with AutoCAD via a lightweight plugin, extracting essential data such as floor area, window placement, wall characteristics, and insulation details directly from 2D DWG files."
          />
        </ListItem>
        <Divider component="li" />
        <ListItem>
          <ListItemText
            primary="Machine Learning Model"
            secondary="We employ advanced machine learning techniques to predict energy efficiency metrics like Energy Use Intensity (EUI) and LEED ratings, providing actionable insights for optimizing building designs."
          />
        </ListItem>
        <Divider component="li" />
        <ListItem>
          <ListItemText
            primary="Visualization Tools"
            secondary="The tool offers intuitive visualizations, including heat maps, that highlight areas of inefficiency directly on the blueprints within the AutoCAD environment, facilitating easy identification of potential improvements."
          />
        </ListItem>
        <Divider component="li" />
        <ListItem>
          <ListItemText
            primary="Comprehensive Recommendations"
            secondary="Based on the analysis, the application offers specific recommendations to enhance energy efficiency, such as optimizing window placement or improving insulation."
          />
        </ListItem>
      </List>

      <Typography variant="h6" gutterBottom>
        Integration and Workflow
      </Typography>

      <Typography variant="body2" paragraph>
        Our tool is designed to integrate seamlessly with existing AutoCAD workflows, minimizing disruption while maximizing productivity. The file-based integration approach ensures compatibility with various CAD software versions, making it accessible to a broad range of users.
      </Typography>

      <Typography variant="h6" gutterBottom>
        Documentation and Support
      </Typography>

      <Typography variant="body2" paragraph>
        We provide detailed documentation, including user manuals and technical guides, to help you set up, use, and maintain the application effectively. For further assistance, our support team is always ready to help you with any queries or issues you may encounter.
      </Typography>

      <Typography variant="body2" paragraph>
        By utilizing our application, architects and engineers can contribute to sustainable development by designing more energy-efficient buildings, ultimately reducing environmental impact.
      </Typography>
    </MainCard>
  );
}