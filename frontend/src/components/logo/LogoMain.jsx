import { useTheme } from '@mui/material/styles';
import logo from 'assets/images/icons/Logo.jpg'; // Adjust this path if necessary

// ==============================|| LOGO COMPONENT ||============================== //

const Logo = () => {
  const theme = useTheme();

  return (
    // Use an image instead of an SVG
    <img src={logo} alt="Logo" width="45" />
  );
};

export default Logo;