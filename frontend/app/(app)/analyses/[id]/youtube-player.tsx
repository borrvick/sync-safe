"use client";

import { useEffect, useRef } from "react";
import Script from "next/script";

declare global {
  interface Window {
    YT: {
      Player: new (
        element: HTMLElement,
        options: {
          videoId: string;
          playerVars?: Record<string, number | string>;
          height?: string;
          width?: string;
          events?: { onReady?: () => void };
        }
      ) => { seekTo: (seconds: number, allowSeekAhead: boolean) => void };
    };
    onYouTubeIframeAPIReady: () => void;
  }
}

function extractVideoId(url: string): string | null {
  try {
    const u = new URL(url);
    if (u.hostname.includes("youtube.com")) return u.searchParams.get("v");
    if (u.hostname === "youtu.be") return u.pathname.slice(1);
    return null;
  } catch {
    return null;
  }
}

type YTPlayerInstance = { seekTo: (seconds: number, allowSeekAhead: boolean) => void };

export function YouTubePlayer({ sourceUrl }: { sourceUrl: string }) {
  const playerRef = useRef<YTPlayerInstance | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const videoId = extractVideoId(sourceUrl);

  useEffect(() => {
    if (!videoId || !containerRef.current) return;

    function init() {
      if (!containerRef.current) return;
      playerRef.current = new window.YT.Player(containerRef.current, {
        videoId: videoId!,
        playerVars: { rel: 0, modestbranding: 1 },
        height: "100%",
        width: "100%",
      });
    }

    if (window.YT?.Player) {
      init();
    } else {
      window.onYouTubeIframeAPIReady = init;
    }

    window.__syncSafeSeek = (seconds: number) => {
      playerRef.current?.seekTo(seconds, true);
    };

    return () => {
      window.__syncSafeSeek = undefined;
    };
  }, [videoId]);

  if (!videoId) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      <Script
        src="https://www.youtube.com/iframe_api"
        strategy="afterInteractive"
      />
      <div
        ref={containerRef}
        className="aspect-video w-full"
        aria-label="YouTube player"
      />
    </div>
  );
}
