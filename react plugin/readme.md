# React Chatbot

A simple React-based chatbot widget.

## Getting Started

1. **Initialize a React Project (with TypeScript)**

   If you haven't already, create a new React project using [Vite](https://vitejs.dev/) or [Create React App](https://create-react-app.dev/) **with TypeScript**:

   ```sh
   npm create vite@latest -- --template react-ts
   # or
   npx create-react-app my-chatbot --template typescript
   ```

2. **Install Dependencies**

   Navigate to your project folder and install dependencies:

   ```sh
   npm install
   ```

3. **Project Structure**

   - Main chatbot logic: [`src/components/chat.tsx`](src/components/chat.tsx)
   - Styles: [`src/components/styles.css`](src/styles/styles.css)

4. **Usage**

   - Import and use the chatbot widget in your main app file (e.g., `App.tsx`):

     ````tsx
     import ChatbotWidget from './components/chat';
     import './styles/styles.css';

     function App() {
       return (
         <div>
           <ChatbotWidget />
         </div>
       );
     }

     export default App;
     ````

5. **Run the Project**

   ```sh
   npm run dev
   # or
   npm start
   ```

## Customization

- Edit [`src/components/chat.tsx`](src/components/chat.tsx) to change chatbot logic or UI.
- Update styles in [`src/styles/styles.css`](src/styles/styles.css).