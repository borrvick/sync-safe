// Global seek bridge set by YouTubePlayer and consumed by TimestampButton.
interface Window {
  __syncSafeSeek?: (seconds: number) => void;
}
