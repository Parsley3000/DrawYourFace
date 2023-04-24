import React, { useState } from "react";
import { Document, Page, StyleSheet, View, Image as PDFImage, Text as PDFText, PDFDownloadLink } from '@react-pdf/renderer';
import {
  Box,
  Button,
  ChakraProvider,
  Container,
  extendTheme,
  Flex,
  FormControl,
  Image,
  VStack,
  Tooltip,
  HStack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  Select,
  Center
} from "@chakra-ui/react";
import Webcam from "react-webcam";
import axios from "axios";

const theme = extendTheme({
  components: {
    FileInput: {
      baseStyle: {
        input: {
          opacity: 0,
          position: "absolute",
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          height: "100%",
          width: "100%",
          cursor: "pointer",
        },
      },
    },
  },
});

const styles = StyleSheet.create({
  container: {
    flexDirection: 'column',
    padding: 10,
  },
  image: {
    width: '100%',
    height: 'auto',
  },
  text: {
    marginTop: 8,
    marginBottom: 16,
  },
});

const RotatedImagesPDF = ({ images, texts }) => (
  <Document>
    <Page size="A4" style={styles.container}>
      {images.map((image, index) => (
        <View key={index}>
          <PDFImage src={image} style={styles.image} />
          <PDFText>{texts[index]}</PDFText>
        </View>
      ))}
    </Page>
  </Document>
);

function Navbar() {
  const navigateToGithub = () => {
    window.location.href = "https://github.com/Parsley3000/DrawYourFace";
  };
  const navigateToStableDiffusion = () => {
    window.location.href = "https://huggingface.co/docs/diffusers/index";
  };
  const navigateToFacialRecognition = () => {
    window.location.href = "https://github.com/codeniko/shape_predictor_81_face_landmarks";
  };
  const navigateToTUDublin = () => {
    window.location.href = "https://www.tudublin.ie/";
  };
  const navigateToSDModels = () => {
    window.location.href = "https://civitai.com/";
  };

  return (
    <Flex
      as="nav"
      align="center"
      justify="center"
      direction="column"
      paddingY={4}
      bg="gray.800"
      color="white"
    >
      <Box>
        <Text fontSize="4xl" textAlign="center">Draw My Face</Text>
      </Box>
      <Box display="flex" alignItems="center" justifyContent="center" marginTop={4}>
        <Button colorScheme='blue' mx={2} flex={1} maxW="120px" onClick={navigateToGithub}>
          Github
        </Button>
        <Button colorScheme='red' mx={2} flex={1} maxW="120px" onClick={navigateToStableDiffusion}>
          Diffusers
        </Button>
        <Button colorScheme='orange' mx={2} flex={1} maxW="120px" onClick={navigateToFacialRecognition}>
          Facial Recog
        </Button>
        <Button colorScheme='yellow' mx={2} flex={1} maxW="120px" onClick={navigateToTUDublin}>
          TUDublin
        </Button>
        <Button colorScheme='green' mx={2} flex={1} maxW="120px" onClick={navigateToSDModels}>
          SD Models
        </Button>
      </Box>
    </Flex>
  );
}

function App() {
  const [leftImage, setLeftImage] = useState(null);
  const [rightImage, setRightImage] = useState(null);
  const [webcamOpen, setWebcamOpen] = useState(false);
  const [isGenerateHovered, setIsGenerateHovered] = useState(false);
  const [isInstructionsHovered, setIsInstructionsHovered] = useState(false);
  const [instructionsGenerated, setInstructionsGenerated] = useState(false);
  const [instructionImages, setInstructionImages] = useState([]);
  const [genderValue, setGenderValue] = useState(2);
  const [ageValue, setAgeValue] = useState(2);
  const [styleValue, setStyleValue] = useState("Anime");
  const [instructionTexts, setInstructionTexts] = useState([]);
  const [apiCallCompleted, setApiCallCompleted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setLeftImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerate = async () => {
    if (leftImage) {
      setIsLoading(true);
      try {
        const response = await axios.post(
          "http://localhost:8000/rotate-image/",
          {
            image: leftImage,
            gender: genderValue, 
            age: ageValue,
            style: styleValue,
          },
          { headers: { "Content-Type": "application/json" } }
        );
        setRightImage(`data:image/png;base64,${response.data.image}`);
      } catch (error) {
        console.error(error);
      } finally {
        setIsLoading(false); // Set isLoading to false after the API call is completed
      }
    }
  };

  const handleInstructions = async () => {
    if (rightImage) {
      setApiCallCompleted(false);
      try {
        const response = await axios.post(
          "http://localhost:8000/instructions/",
          { image: rightImage },
          { headers: { "Content-Type": "application/json" } }
        );
        setInstructionImages(
          response.data.images.map((img) => `data:image/png;base64,${img}`)
        );
        setInstructionTexts(response.data.texts);
        setApiCallCompleted(true); // Set the state to true after a successful API call
      } catch (error) {
        console.error(error);
      }
      console.log(instructionImages);
      console.log(instructionTexts);
    }
  };

  const webcamRef = React.useRef(null);
  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setLeftImage(imageSrc);
    setWebcamOpen(false); //This line is to close the camera after taking a picture
  };

  return (
    <ChakraProvider theme={theme}>
      <Navbar />
      <Container maxW="100%" py={5}>
        {apiCallCompleted && (
          <PDFDownloadLink
            document={<RotatedImagesPDF images={instructionImages} texts={instructionTexts} />}
            fileName="instructions.pdf"
            style={{
              textDecoration: 'none',
              position: 'fixed',
              bottom: '16px',
              left: '16px',
              zIndex: 'docked',
              padding: '8px 16px',
              backgroundColor: '#3ea7ab',
              borderRadius: '4px',
              color: 'white',
            }}
          >
            Download as PDF
          </PDFDownloadLink>
        )}
        <Flex alignItems="center" justifyContent="center" flexDirection="column">
        <HStack>
          <Box>
            <VStack spacing={5}> 
              <HStack spacing="50">
                <VStack spacing="20px">
                  <Box>
                    <Text fontSize="lg" textAlign="center">
                      Gender
                    </Text>
                    <Slider
                      aria-label="Gender slider"
                      width="300px"
                      min={0}
                      max={4}
                      step={1}
                      defaultValue={2}
                      onChange={(value) => setGenderValue(value)}
                    >
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <HStack justifyContent="space-between">
                      <Text fontSize="sm">Male</Text>
                      <Text fontSize="sm">Neutral</Text>
                      <Text fontSize="sm">Female</Text>
                    </HStack>
                  </Box>
                  <Box>
                    <Text fontSize="lg" textAlign="center">
                      Age
                    </Text>
                    <Slider
                      aria-label="Gender slider"
                      width="300px"
                      min={0}
                      max={4}
                      step={1}
                      defaultValue={2}
                      onChange={(value) => setAgeValue(value)}
                    >
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <HStack justifyContent="space-between">
                      <Text fontSize="sm">Very Young</Text>
                      <Text fontSize="sm">Middle Aged</Text>
                      <Text fontSize="sm">Very Old</Text>
                    </HStack>
                  </Box>
                </VStack>
                <Box>
                  <Text fontSize="lg" textAlign="center">
                    Style
                  </Text>
                  <Select
                    value={styleValue}
                    onChange={(event) => setStyleValue(event.target.value)}
                  >
                    <option value="Anime">Anime</option>
                    <option value="Cartoon">Cartoon</option>
                    <option value="Sketch">Sketch</option>
                  </Select>
                </Box>
              </HStack>
              {webcamOpen ? (
                <Box>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/png"
                    width={512}
                    height={512}
                    screenshotWidth={512}
                    screenshotHeight={512}
                  />
                  <Button onClick={capture} mt={2} colorScheme='yellow'>
                    Take Picture
                  </Button>
                  <Button onClick={() => setWebcamOpen(false)} mt={2} colorScheme='yellow'>
                    Close Camera
                  </Button>
                </Box>
              ) : (
                <Box position="relative" width="512px" height="512px">
                  <Image
                    boxSize="512px"
                    objectFit="cover"
                    border="2px"
                    src={leftImage || "https://i.pinimg.com/564x/82/50/eb/8250ebbe710fdc11dc3332e02ad7cf42.jpg"}
                    alt="Left Image"
                  />
                  <FormControl>
                  <label htmlFor="file-upload" style={{ position: "absolute", cursor: "pointer" }}>
                    <Button as="span" size="sm">Upload Image</Button>
                  </label>
                  <input
                    type="file"
                    id="file-upload"
                    accept=".png,.jpg,.jpeg"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                  />
                </FormControl>
                </Box>
              )}
              <Button onClick={() => setWebcamOpen(true)} colorScheme='yellow'>Camera</Button>
            </VStack>
          </Box>
          <Box padding="100">
            <Tooltip
              label="Upload an image"
              isOpen={!leftImage && isGenerateHovered}
              hasArrow
            >
              <Button
                onClick={() => {
                  handleGenerate();
                  setInstructionsGenerated(true);
                }}
                disabled={!leftImage || webcamOpen}
                mb={5}
                onMouseEnter={() => setIsGenerateHovered(true)}
                onMouseLeave={() => setIsGenerateHovered(false)}
                colorScheme='yellow'
              >
                Generate
              </Button>
            </Tooltip>
          </Box>
          <Box paddingTop="28">
            <Image
              boxSize="512px"
              objectFit="cover"
              src={
                isLoading
                  ? "https://cdn.dribbble.com/users/1787505/screenshots/7300251/media/a351d9e0236c03a539181b95faced9e0.gif"
                  : rightImage || "http://clipart-library.com/images/pio5xq6BT.jpg"
              }
              alt="Right Image"
            />
          </Box>
        </HStack>
          <Tooltip
            label="Generate an image"
            isOpen={!instructionsGenerated && isInstructionsHovered}
            hasArrow
          >
            <Button
              onClick={handleInstructions}
              disabled={!instructionsGenerated}
              mb={5}
              onMouseEnter={() => setIsInstructionsHovered(true)}
              onMouseLeave={() => setIsInstructionsHovered(false)}
              colorScheme='yellow'
            >
              Instructions
            </Button>
          </Tooltip>
          <VStack spacing={6} marginBottom={5}>
            {instructionImages.map((image, index) => (
              <Box key={index}>
                <Center>
                  <Image boxSize="512px" objectFit="cover" src={image} alt={`Instruction Images ${index + 1}`} />
                </Center>
                <Center width="550px" paddingTop="15px" textAlign="center">
                  <Text fontSize='20px'>{instructionTexts[index]}</Text> {/* Add this line to display the text below the image */}
                </Center>
              </Box>
            ))}
          </VStack>
        </Flex>
      </Container>
    </ChakraProvider>
  );
}

export default App;