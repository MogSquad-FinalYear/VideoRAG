"""
VideoRAG — Agent Tool Definitions
Tool schemas for Groq function calling.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_caption",
            "description": "Search for video frames by their visual description/caption. Use this when the user asks about visual content, objects, scenes, or activities seen in the video. Returns matching frames with captions and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what to search for (e.g., 'person holding a bag', 'red car on the street')"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within. Omit to search all videos."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_visual_similarity",
            "description": "Search for visually similar frames using CLIP image embeddings. Use this when you need to find frames that look similar to a text description based on visual features (colors, shapes, objects). More precise for visual matching than caption search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text description of the visual content to find"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": "Search through speech transcripts of videos. Use this when the user asks about something that was SAID or SPOKEN in the video, or when looking for specific words, phrases, or dialog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in speech transcripts"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_image",
            "description": "Search for frames visually similar to a reference image. Use this for image-to-video retrieval — when a user provides a reference image and wants to find similar-looking frames in the videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "File path to the reference image to search for"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_clip_info",
            "description": "Get information about a specific time range in a video, including the frames in that range. Use this to retrieve details about a specific moment identified by other search tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The video ID"
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds"
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds"
                    }
                },
                "required": ["video_id", "start_time", "end_time"]
            }
        }
    }
]
