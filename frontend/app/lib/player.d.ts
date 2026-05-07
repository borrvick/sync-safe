// Typed detail payload for the seek-to-seconds CustomEvent fired by TimestampButton
// and handled by YouTubePlayer. Using a CustomEvent instead of a window global
// avoids shared mutable state and prevents naming conflicts with third-party scripts.
interface SeekToSecondsEventDetail {
  seconds: number;
}
