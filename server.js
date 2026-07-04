import express from "express";
import dotenv from "dotenv";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import path from "path";  // use to find path

dotenv.config();
const port = 3000;
const app = express();
app.use(express.json());

const __dirname = path.resolve();
app.use(express.static(path.join(__dirname, "public")));

// model
const model = new ChatGoogleGenerativeAI({
    model: "models/gemini-2.5-flash",
    maxOutputTokens: 2048,
    apiKey: process.env.GOOGLE_API_KEY,
});

// tools
const getMenuTool = new DynamicStructuredTool({
    name: "getMenuTool",
    description: "Returns the final answer for today's menu for the given category (breakfast, lunch, or dinner). Use this tool directly answer the user's menu question",
    schema: z.object({
        category: z.string().describe("Type of food. Example: breakfast, lunch, dinner"),
    }),
    func: async ({ category }) => {
        const menus = {
            breakfast: "Aaluu Paratha, Poha, Masala Tea",
            lunch: "Panner butter Masala, Dal Fry, Jerra Rice, Roti",
            dinner: "Veg Biryani, Raita, Salad, Gulab Jamun",
        };
        return menus[category.toLowerCase()] || "No Menu found for this category.";
    },
});

// prompt
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant. Use your tools to look up the menu, and then directly present the tool's observation as your final response to the user. Do not call the tool again if you already have the menu."],
    ["human", "{input}"],
    ["ai", "{agent_scratchpad}"]
]);

// Agent (This one is async)
const agent = await createToolCallingAgent({
    llm: model,
    tools: [getMenuTool],
    prompt
});

// Executor 
const executor = AgentExecutor.fromAgentAndTools({
    agent,
    tools: [getMenuTool],
    verbose: true,
    maxIterations: 3,
    returnIntermediateSteps: true
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API 
app.post("/api/chat", async (req, res) => {
    const userInput = req.body.input;
    console.log("userInput : ", userInput);

    try {
        const response = await executor.invoke({ input: userInput });
        console.log("Agent full Response: ", response);
        
        const data = response.intermediateSteps?.[0]?.observation;
        
        if (response.output && response.output !== "Agent stopped due to max iteration.") {
            return res.json({ output: response.output });
        } else if (data != null) {
            return res.json({ output: data });
        }
        res.status(500).json({ output: "Agent couldn't find a valid answer." });
    }
    catch (err) {
        console.error("Error During Agent Execution", err);
        res.status(500).json({ output: "Sorry, something went wrong. Please try again." });
    }
});

// Server Listen 
app.listen(port, () => {
    console.log(`Server is running on port http://localhost:${port}`);
});